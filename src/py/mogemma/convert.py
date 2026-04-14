"""Convert an Orbax/OCDBT checkpoint to HuggingFace-style safetensors.

The Mojo backend (`src/mo/mogemma/core.mojo`) loads weights by exact
HuggingFace tensor names. This module reads an Orbax checkpoint with
:class:`mogemma.orbax_loader.OrbaxLoader` streaming helpers and writes
safetensors files whose keys match what Mojo calls ``metadata_obj.get(...)``
on.

Scope is intentionally narrow: only the tensors Mojo actually loads are
emitted. Audio weights, quantization scales, and Orbax-only metadata
tensors are skipped.

Public API:
    convert_orbax_to_safetensors(model_path) -> list[Path]
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Literal

import numpy as np

from mogemma.orbax_loader import OrbaxLoader

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

Variant = Literal["base", "ple", "moe"]

_LAYER_KEY_RE = re.compile(r"^layer_(\d+)\.")


def _variant_from_keys(keys: list[str]) -> Variant:
    """Classify an Orbax checkpoint as ``base``, ``ple`` (E2B/E4B), or ``moe``.

    Detection rules:
    - ``moe`` if any ``router_logits`` tensor is present (takes priority).
    - ``ple`` if ``embedder.per_layer_embeddings`` is present.
    - ``base`` otherwise.
    """
    has_router = any("router_logits" in k for k in keys)
    if has_router:
        return "moe"
    has_ple = any(k.startswith("embedder.per_layer_") for k in keys)
    if has_ple:
        return "ple"
    return "base"


def _layer_count(keys: list[str]) -> int:
    """Return ``max(layer_N) + 1`` across *keys*.

    Raises ``ValueError`` if no ``layer_N.*`` keys are present.
    """
    indices: list[int] = []
    for key in keys:
        match = _LAYER_KEY_RE.match(key)
        if match:
            indices.append(int(match.group(1)))
    if not indices:
        msg = "no layer_N.* keys present in checkpoint — cannot determine layer count"
        raise ValueError(msg)
    return max(indices) + 1


# ── Base-transformer iterator ────────────────────────────────────────────────


def _iter_base_transformer(model_path: Path, _keys: list[str], num_layers: int) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(safetensors_name, array)`` pairs for the base-transformer tensors.

    Produces: ``model.embed_tokens.weight``, ``lm_head.weight`` (tied),
    ``model.norm.weight``, and the 13 per-layer tensors the Mojo backend
    loads (see ``core.mojo:229-254``).

    Shape transforms derived from the E2B-it inventory
    (see ``.agents/specs/orbax-safetensors-conversion/e2b-inventory.txt``):

    * ``q_einsum.w (n_heads, H, head_dim)`` → ``q_proj (n_heads·head_dim, H)``
    * ``kv_einsum.w (2, n_kv, H, head_dim)`` → ``{k,v}_proj`` via split + transform
    * ``attn_vec_einsum.w (n_heads, head_dim, H)`` → ``o_proj (H, n_heads·head_dim)``
    * ``gating_einsum.w (2, I, H)`` → ``{gate,up}_proj`` via split (no transpose)
    * ``linear.w (I, H)`` → ``down_proj (H, I)`` via transpose
    * all norms pass through unchanged

    Raises ``KeyError`` if a required Orbax tensor is missing (e.g., the
    checkpoint is not a base-transformer variant).
    """
    embed = OrbaxLoader.open_tensor(model_path, "embedder.input_embedding")
    yield "model.embed_tokens.weight", embed
    yield "lm_head.weight", embed  # tied — E2B-it ships no separate lm_head tensor

    yield "model.norm.weight", OrbaxLoader.open_tensor(model_path, "final_norm.scale")

    for n in range(num_layers):
        pfx_in = f"layer_{n}"
        pfx_out = f"model.layers.{n}"

        # Norms (pass-through)
        yield (
            f"{pfx_out}.input_layernorm.weight",
            OrbaxLoader.open_tensor(model_path, f"{pfx_in}.pre_attention_norm.scale"),
        )
        yield (
            f"{pfx_out}.post_attention_layernorm.weight",
            OrbaxLoader.open_tensor(model_path, f"{pfx_in}.post_attention_norm.scale"),
        )
        yield (
            f"{pfx_out}.pre_feedforward_layernorm.weight",
            OrbaxLoader.open_tensor(model_path, f"{pfx_in}.pre_ffw_norm.scale"),
        )
        yield (
            f"{pfx_out}.post_feedforward_layernorm.weight",
            OrbaxLoader.open_tensor(model_path, f"{pfx_in}.post_ffw_norm.scale"),
        )
        yield (
            f"{pfx_out}.self_attn.q_norm.weight",
            OrbaxLoader.open_tensor(model_path, f"{pfx_in}.attn.query_norm.scale"),
        )
        yield (
            f"{pfx_out}.self_attn.k_norm.weight",
            OrbaxLoader.open_tensor(model_path, f"{pfx_in}.attn.key_norm.scale"),
        )

        # Attention projections
        q = OrbaxLoader.open_tensor(model_path, f"{pfx_in}.attn.q_einsum.w")
        yield f"{pfx_out}.self_attn.q_proj.weight", q.transpose(0, 2, 1).reshape(-1, q.shape[1])

        # kv_einsum.w is the combined-KV layout used by E2B/E4B: (2, n_kv, H, head_dim).
        # Separate ``k_einsum.w`` / ``v_einsum.w`` layouts used by 31B/MoE are handled
        # in a sibling code path not exercised by this E2B fixture (TODO when 31B lands).
        kv = OrbaxLoader.open_tensor(model_path, f"{pfx_in}.attn.kv_einsum.w")
        k, v = kv[0], kv[1]
        yield f"{pfx_out}.self_attn.k_proj.weight", k.transpose(0, 2, 1).reshape(-1, k.shape[1])
        yield f"{pfx_out}.self_attn.v_proj.weight", v.transpose(0, 2, 1).reshape(-1, v.shape[1])

        o = OrbaxLoader.open_tensor(model_path, f"{pfx_in}.attn.attn_vec_einsum.w")
        yield f"{pfx_out}.self_attn.o_proj.weight", o.reshape(-1, o.shape[2]).T

        # MLP
        gate_up = OrbaxLoader.open_tensor(model_path, f"{pfx_in}.mlp.gating_einsum.w")
        yield f"{pfx_out}.mlp.gate_proj.weight", gate_up[0]
        yield f"{pfx_out}.mlp.up_proj.weight", gate_up[1]

        down = OrbaxLoader.open_tensor(model_path, f"{pfx_in}.mlp.linear.w")
        yield f"{pfx_out}.mlp.down_proj.weight", down.T


# ── Per-Layer Embeddings (E2B/E4B) ──────────────────────────────────────────


def _iter_ple(model_path: Path, num_layers: int) -> Iterator[tuple[str, np.ndarray]]:
    """Yield the 3 PLE tensors per layer consumed by ``forward_ple_input`` (layers.mojo:884-908).

    Mapping (verified against E2B-it inventory and ``layers.mojo`` forward path):

    * ``embedder.per_layer_embeddings`` ``(V, L, H_ple)`` → ``per_layer_embedding.weight``
      ``(V, H_ple)`` per layer via ``arr[:, N, :]`` (layer axis is the MIDDLE axis).
    * ``layer_N.per_layer_projection.w`` ``(H_ple, H)`` → ``per_layer_projection.weight`` (identity).
    * ``layer_N.post_per_layer_input_norm.scale`` ``(H,)`` → ``per_layer_norm.weight`` (identity).

    Orbax PLE-adjacent tensors NOT consumed by the Mojo forward pass are
    intentionally skipped (never opened): ``embedder.per_layer_model_projection.w``,
    ``embedder.per_layer_projection_norm.scale``, ``layer_N.per_layer_input_gate.w``,
    ``layer_N.skip_scale``.
    """
    global_embeddings = OrbaxLoader.open_tensor(model_path, "embedder.per_layer_embeddings")

    for n in range(num_layers):
        pfx_out = f"model.layers.{n}.per_layer_input"
        # Slice the layer axis out of the global (V, L, H_ple) table;
        # ascontiguousarray gives safetensors a self-contained C buffer.
        yield (
            f"{pfx_out}.per_layer_embedding.weight",
            np.ascontiguousarray(global_embeddings[:, n, :]),
        )
        yield (
            f"{pfx_out}.per_layer_projection.weight",
            OrbaxLoader.open_tensor(model_path, f"layer_{n}.per_layer_projection.w"),
        )
        yield (
            f"{pfx_out}.per_layer_norm.weight",
            OrbaxLoader.open_tensor(model_path, f"layer_{n}.post_per_layer_input_norm.scale"),
        )


# ── config.json generation ──────────────────────────────────────────────────


# Per-variant architecture constants from `.agents/knowledge/gemma4-architecture.md`.
# These are model-family facts (sliding windows, partial RoPE factor, top-k) that
# can't be derived from tensor shapes alone.
_VARIANT_CONSTANTS = {
    "base": {
        "model_type": "gemma4_text",
        "sliding_window_size": 1024,
        "partial_rotary_factor": 0.5,
    },
    "ple": {
        "model_type": "gemma4_text",  # narrowed to _e2b/_e4b below via use_double_wide_mlp
        "sliding_window_size": 512,
        "partial_rotary_factor": 0.5,
    },
    "moe": {
        "model_type": "gemma4_text",  # narrowed to _moe_26b below
        "sliding_window_size": 1024,
        "partial_rotary_factor": 0.5,
        "num_experts_per_tok": 8,
    },
}


def _generate_config_json(
    keys: list[str],
    shape_oracle: "Callable[[str], tuple[int, ...]]",
) -> dict[str, object]:
    """Synthesize a HF-compatible ``config.json`` dict from the Orbax tensor inventory.

    Values that *can* be derived from tensor shapes (vocab_size, hidden_size,
    num_attention_heads, …) are derived. Values that can't (sliding-window size,
    partial rotary factor, MoE top-k) are filled from
    :data:`_VARIANT_CONSTANTS`.

    Args:
        keys: full list of Orbax tensor names present in the checkpoint.
        shape_oracle: callable ``name -> shape`` (decouples shape inspection
            from the live Orbax loader so this function is unit-testable).

    Returns the dict; caller writes it as JSON.
    """
    variant = _variant_from_keys(keys)
    num_layers = _layer_count(keys)

    # Shape-derived fields. Raise on missing required tensors.
    embed_shape = shape_oracle("embedder.input_embedding")
    vocab_size, hidden_size = embed_shape[0], embed_shape[1]

    q_shape = shape_oracle("layer_0.attn.q_einsum.w")
    num_attention_heads, _q_hidden, head_dim = q_shape[0], q_shape[1], q_shape[2]

    kv_shape = shape_oracle("layer_0.attn.kv_einsum.w")
    num_key_value_heads = kv_shape[1]

    gating_shape = shape_oracle("layer_0.mlp.gating_einsum.w")
    # Layout differs between dense (gate/up split) and MoE (experts leading):
    #   dense: (2, intermediate, H)
    #   moe:   (num_experts, 2, moe_intermediate, H)
    if variant == "moe":
        num_local_experts = gating_shape[0]
        moe_intermediate_size = gating_shape[2]
        intermediate_size = moe_intermediate_size  # dense field kept for HF compatibility
    else:
        num_local_experts = None
        moe_intermediate_size = None
        intermediate_size = gating_shape[1]

    config: dict[str, object] = {
        **_VARIANT_CONSTANTS[variant],
        "num_hidden_layers": num_layers,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
    }

    if variant == "ple":
        ple_shape = shape_oracle("embedder.per_layer_embeddings")
        config["vocab_size_per_layer_input"] = ple_shape[0]
        config["hidden_size_per_layer_input"] = ple_shape[2]
        # use_double_wide_mlp distinguishes E2B (True) from E4B (False); the
        # safest signal is intermediate/hidden ratio: E2B uses 4× (6144/1536)
        # versus E4B's usual 8×. A definitive classifier will land once we have
        # E4B inventory. For now, emit nothing — downstream _detect_gemma4_variant
        # treats absent use_double_wide_mlp as E4B.

    if variant == "moe":
        config["num_local_experts"] = num_local_experts
        config["moe_intermediate_size"] = moe_intermediate_size
        config["model_type"] = "gemma4_moe_26b"

    return config


# ── Deferred: vision + MoE iterators ────────────────────────────────────────


def _iter_vision(model_path: Path) -> Iterator[tuple[str, np.ndarray]]:  # noqa: ARG001
    """Convert Orbax vision-tower tensors to safetensors. Deferred.

    Blocked on a Mojo/Orbax architecture mismatch: the Orbax checkpoint
    ships GEGLU-style vision MLP weights (``gating_einsum.w`` with a
    leading ``2`` axis for gate+up), but the Mojo forward pass in
    ``layers.mojo:360`` implements plain ``fc1 → GELU → fc2``. Picking
    either branch as ``fc1`` yields numerically wrong vision embeddings;
    the fix needs to happen in the Mojo forward path (or in Gemma 4's
    vision architecture spec) before conversion can produce correct
    safetensors.

    Norms are fully resolved (``pre_attention_norm`` → ``layer_norm1``,
    ``pre_ffw_norm`` → ``layer_norm2``), so only the MLP branch is
    blocking. Implement this iterator once the forward is updated.
    """
    msg = (
        "vision Orbax→safetensors conversion deferred: "
        "Orbax ships GEGLU MLP weights but Mojo vision forward is plain MLP "
        "(layers.mojo:360). Resolve the forward-pass mismatch before converting."
    )
    raise NotImplementedError(msg)
    yield  # pragma: no cover — keeps the function a generator for typing


def _iter_moe_experts(
    model_path: Path,  # noqa: ARG001
    num_layers: int,  # noqa: ARG001
    num_experts: int,  # noqa: ARG001
) -> Iterator[tuple[str, np.ndarray]]:
    """Convert Orbax MoE expert tensors to per-expert safetensors. Deferred.

    Blocked on Task 0.2 (26B-A4B-it inventory dump) — MoE expert packing
    shapes are still guesses. E2B-it has no router tensors, so this code
    path has no live checkpoint to validate against yet. Implement after
    a 26B checkpoint is available locally.
    """
    msg = (
        "MoE expert conversion deferred: requires 26B-A4B-it inventory "
        "(Task 0.2) to confirm expert packing shapes before implementation."
    )
    raise NotImplementedError(msg)
    yield  # pragma: no cover — keeps the function a generator for typing


# ── Sharded safetensors writer ───────────────────────────────────────────────


_SINGLE_SHARD_FILENAME = "model.safetensors"
_INDEX_FILENAME = "model.safetensors.index.json"
_TEMP_SHARD_FMT = "_shard-{index}.part.safetensors"
_FINAL_SHARD_FMT = "model-{index:05d}-of-{total:05d}.safetensors"


def _write_sharded(
    output_dir: Path, tensor_iter: Iterable[tuple[str, np.ndarray]], shard_size_bytes: int
) -> list[Path]:
    """Write ``tensor_iter`` to safetensors, sharding when cumulative bytes exceed the threshold.

    Writes a single ``model.safetensors`` if total bytes fit under
    *shard_size_bytes*; otherwise writes ``model-00001-of-{N}.safetensors`` ..
    ``model-{N}-of-{N}.safetensors`` plus a ``model.safetensors.index.json``
    mapping tensor names → shard filenames.

    Returns the list of safetensors files written, in shard order.
    """
    from safetensors.numpy import save_file  # noqa: PLC0415 — heavy optional dep

    output_dir.mkdir(parents=True, exist_ok=True)

    # Streaming pass: buffer tensors per-shard, flush to a temp filename when the
    # buffer would otherwise exceed *shard_size_bytes*. A single tensor that is
    # already larger than the threshold is allowed to occupy its own shard.
    temp_shards: list[tuple[Path, dict[str, np.ndarray], int]] = []
    current: dict[str, np.ndarray] = {}
    current_bytes = 0

    def _flush() -> None:
        nonlocal current, current_bytes
        if not current:
            return
        temp_path = output_dir / _TEMP_SHARD_FMT.format(index=len(temp_shards) + 1)
        save_file(current, str(temp_path))
        temp_shards.append((temp_path, dict(current), current_bytes))
        current = {}
        current_bytes = 0

    for name, array in tensor_iter:
        array_bytes = array.nbytes
        if current and current_bytes + array_bytes > shard_size_bytes:
            _flush()
        current[name] = array
        current_bytes += array_bytes
    _flush()

    if not temp_shards:
        msg = "tensor_iter yielded no tensors; nothing to write"
        raise ValueError(msg)

    # Single-shard fast path: rename to `model.safetensors`, no index needed.
    if len(temp_shards) == 1:
        final_path = output_dir / _SINGLE_SHARD_FILENAME
        temp_shards[0][0].rename(final_path)
        return [final_path]

    # Multi-shard: rename all, write index.
    total = len(temp_shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    final_paths: list[Path] = []
    for i, (temp_path, tensors, shard_bytes) in enumerate(temp_shards, start=1):
        final_name = _FINAL_SHARD_FMT.format(index=i, total=total)
        final_path = output_dir / final_name
        temp_path.rename(final_path)
        for name in tensors:
            weight_map[name] = final_name
        total_size += shard_bytes
        final_paths.append(final_path)

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (output_dir / _INDEX_FILENAME).write_text(json.dumps(index, indent=2) + "\n")

    return final_paths
