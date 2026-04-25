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
from typing import TYPE_CHECKING, Literal

import numpy as np

from mogemma.orbax_loader import OrbaxLoader

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
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


# ── Attention iterator helpers ───────────────────────────────────────────────


def _iter_attn_layer(model_path: Path, keys_set: set[str], layer_n: int) -> Iterator[tuple[str, np.ndarray]]:
    """Yield the 6 attention tensors for a transformer layer.

    Handles three Orbax KV layouts observed across variants:

    * ``kv_einsum.w`` ``(2, n_kv, H, head_dim)`` — combined k+v (E2B/E4B + MoE
      sliding layers). Split along axis 0.
    * ``k_einsum.w`` + ``v_einsum.w`` — separate k/v tensors.
    * ``k_einsum.w`` alone (no ``v_einsum.w``) — full-attention layers with
      ``attention_k_eq_v`` / KV-sharing; emit k_proj and v_proj as the same
      transformed array.

    Common reshape for q/k/v: ``w.transpose(0, 2, 1).reshape(-1, H)`` so the
    output is ``[n_heads·head_dim, H]`` matching the HF convention.
    """
    pfx_in = f"layer_{layer_n}"
    pfx_out = f"model.layers.{layer_n}"

    q = OrbaxLoader.open_tensor(model_path, f"{pfx_in}.attn.q_einsum.w")
    yield f"{pfx_out}.self_attn.q_proj.weight", q.transpose(0, 2, 1).reshape(-1, q.shape[1])

    kv_combined = f"{pfx_in}.attn.kv_einsum.w"
    k_only = f"{pfx_in}.attn.k_einsum.w"
    v_only = f"{pfx_in}.attn.v_einsum.w"

    if kv_combined in keys_set:
        kv = OrbaxLoader.open_tensor(model_path, kv_combined)
        k, v = kv[0], kv[1]
        yield f"{pfx_out}.self_attn.k_proj.weight", k.transpose(0, 2, 1).reshape(-1, k.shape[1])
        yield f"{pfx_out}.self_attn.v_proj.weight", v.transpose(0, 2, 1).reshape(-1, v.shape[1])
    elif k_only in keys_set:
        k = OrbaxLoader.open_tensor(model_path, k_only)
        k_proj = k.transpose(0, 2, 1).reshape(-1, k.shape[1])
        yield f"{pfx_out}.self_attn.k_proj.weight", k_proj
        if v_only in keys_set:
            v = OrbaxLoader.open_tensor(model_path, v_only)
            yield f"{pfx_out}.self_attn.v_proj.weight", v.transpose(0, 2, 1).reshape(-1, v.shape[1])
        else:
            # KV-sharing (attention_k_eq_v): v_proj reuses the k projection.
            yield f"{pfx_out}.self_attn.v_proj.weight", k_proj
    else:
        msg = f"layer {layer_n}: no kv_einsum.w or k_einsum.w found in Orbax inventory"
        raise KeyError(msg)

    o = OrbaxLoader.open_tensor(model_path, f"{pfx_in}.attn.attn_vec_einsum.w")
    yield f"{pfx_out}.self_attn.o_proj.weight", o.reshape(-1, o.shape[2]).T

    yield (f"{pfx_out}.self_attn.q_norm.weight", OrbaxLoader.open_tensor(model_path, f"{pfx_in}.attn.query_norm.scale"))
    yield (f"{pfx_out}.self_attn.k_norm.weight", OrbaxLoader.open_tensor(model_path, f"{pfx_in}.attn.key_norm.scale"))


# ── Base-transformer iterator ────────────────────────────────────────────────


def _iter_base_transformer(model_path: Path, keys: list[str], num_layers: int) -> Iterator[tuple[str, np.ndarray]]:
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

    keys_set = set(keys)

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

        # Attention (q/k/v/o + q_norm/k_norm) — delegates KV-layout detection.
        yield from _iter_attn_layer(model_path, keys_set, n)

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
        yield (f"{pfx_out}.per_layer_embedding.weight", np.ascontiguousarray(global_embeddings[:, n, :]))
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
        "final_logit_softcapping": 30.0,
        "attn_logit_softcapping": 50.0,
    },
    "ple": {
        "model_type": "gemma4_text",  # narrowed to _e2b/_e4b below via use_double_wide_mlp
        "sliding_window_size": 512,
        "partial_rotary_factor": 0.5,
        "final_logit_softcapping": 30.0,
        "attn_logit_softcapping": 50.0,
    },
    "moe": {
        "model_type": "gemma4_text",  # narrowed to _moe_26b below
        "sliding_window_size": 1024,
        "partial_rotary_factor": 0.5,
        "final_logit_softcapping": 30.0,
        "attn_logit_softcapping": 50.0,
        "num_experts_per_tok": 8,
    },
}


def _generate_config_json(keys: list[str], shape_oracle: Callable[[str], tuple[int, ...]]) -> dict[str, object]:
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
    # Dense layout is (2, intermediate, H); MoE layout is (num_experts, 2, moe_intermediate, H).
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
        # use_double_wide_mlp distinguishes E2B (True) from E4B (False) via the
        # intermediate/hidden ratio (E2B 4x at 6144/1536 vs E4B 8x). A definitive
        # classifier lands once we have E4B inventory; until then emit nothing and
        # let downstream _detect_gemma4_variant treat absent as E4B.

    if variant == "moe":
        config["num_local_experts"] = num_local_experts
        config["moe_intermediate_size"] = moe_intermediate_size
        config["model_type"] = "gemma4_moe_26b"

    return config


# ── Deferred: vision + MoE iterators ────────────────────────────────────────


def _iter_vision(model_path: Path, num_vision_layers: int) -> Iterator[tuple[str, np.ndarray]]:
    """Yield the vision-tower tensors consumed by ``forward_vision_encoder`` / ``forward_vision_layer``.

    Orbax ``vision_encoder.transformer.stacked_layers.block.*`` tensors are
    vmapped — the leading axis is the vision-layer index. We split along
    that axis and emit per-layer safetensors.

    Gemma 4 vision is GEGLU (as of the 2026-04 Mojo update in ``layers.mojo``):
    ``gating_einsum.w`` has a ``2`` axis for gate+up, which map to
    ``mlp.fc1.weight`` and ``mlp.fc1_up.weight`` respectively.

    Produced tensors (per vision layer I):

    - ``vision_tower.vision_model.embeddings.patch_embedding.weight`` (global)
    - ``vision_tower.vision_model.embeddings.position_embedding.weight`` (global)
    - ``vision_tower.vision_model.post_layernorm.weight`` (derived from last layer's ``post_ffw_norm``)
    - ``multi_modal_projector.linear.weight`` (from ``embedder.mm_input_projection.w``)
    - ``...encoder.layers.I.self_attn.{q,k,v,out}_proj.weight``
    - ``...encoder.layers.I.mlp.{fc1, fc1_up, fc2}.weight``  (GEGLU gate, up, down)
    - ``...encoder.layers.I.layer_norm{1,2}.weight``
    """
    # Global tensors (not stacked)
    patch = OrbaxLoader.open_tensor(model_path, "vision_encoder.entry.input_projection.w")
    yield "vision_tower.vision_model.embeddings.patch_embedding.weight", patch

    pos = OrbaxLoader.open_tensor(model_path, "vision_encoder.entry.pos_emb")
    yield "vision_tower.vision_model.embeddings.position_embedding.weight", pos

    mm = OrbaxLoader.open_tensor(model_path, "embedder.mm_input_projection.w")
    yield "multi_modal_projector.linear.weight", mm

    # Per-layer stacked tensors: leading axis = layer index.
    pfx = "vision_encoder.transformer.stacked_layers.block"
    q_all = OrbaxLoader.open_tensor(model_path, f"{pfx}.attn.q_einsum.w")  # (L, n_heads, H, head_dim)
    kv_all = OrbaxLoader.open_tensor(model_path, f"{pfx}.attn.kv_einsum.w")  # (L, 2, n_kv, H, head_dim)
    o_all = OrbaxLoader.open_tensor(model_path, f"{pfx}.attn.attn_vec_einsum.w")  # (L, n_heads, head_dim, H)
    gate_up_all = OrbaxLoader.open_tensor(model_path, f"{pfx}.mlp.gating_einsum.w")  # (L, 2, intermediate, H)
    linear_all = OrbaxLoader.open_tensor(model_path, f"{pfx}.mlp.linear.w")  # (L, intermediate, H)
    ln1_all = OrbaxLoader.open_tensor(model_path, f"{pfx}.pre_attention_norm.scale")  # (L, H)
    ln2_all = OrbaxLoader.open_tensor(model_path, f"{pfx}.pre_ffw_norm.scale")  # (L, H)
    post_ffw_all = OrbaxLoader.open_tensor(model_path, f"{pfx}.post_ffw_norm.scale")  # (L, H)

    for i in range(num_vision_layers):
        pfx_out = f"vision_tower.vision_model.encoder.layers.{i}"

        q = q_all[i]  # (n_heads, H, head_dim)
        yield f"{pfx_out}.self_attn.q_proj.weight", np.ascontiguousarray(q.transpose(0, 2, 1).reshape(-1, q.shape[1]))

        kv = kv_all[i]  # (2, n_kv, H, head_dim)
        k, v = kv[0], kv[1]
        yield f"{pfx_out}.self_attn.k_proj.weight", np.ascontiguousarray(k.transpose(0, 2, 1).reshape(-1, k.shape[1]))
        yield f"{pfx_out}.self_attn.v_proj.weight", np.ascontiguousarray(v.transpose(0, 2, 1).reshape(-1, v.shape[1]))

        o = o_all[i]  # (n_heads, head_dim, H)
        yield f"{pfx_out}.self_attn.out_proj.weight", np.ascontiguousarray(o.reshape(-1, o.shape[2]).T)

        gate_up = gate_up_all[i]  # (2, intermediate, H)
        yield f"{pfx_out}.mlp.fc1.weight", np.ascontiguousarray(gate_up[0])
        yield f"{pfx_out}.mlp.fc1_up.weight", np.ascontiguousarray(gate_up[1])

        linear = linear_all[i]  # (intermediate, H)
        yield f"{pfx_out}.mlp.fc2.weight", np.ascontiguousarray(linear.T)

        yield f"{pfx_out}.layer_norm1.weight", np.ascontiguousarray(ln1_all[i])
        yield f"{pfx_out}.layer_norm2.weight", np.ascontiguousarray(ln2_all[i])

    # post_layernorm is a Mojo-expected global tensor. The Orbax checkpoint
    # doesn't ship a dedicated top-level post-norm; use the LAST layer's
    # post_ffw_norm as a surrogate (best available signal — matches the
    # position in the forward pass where Mojo applies post_layernorm).
    yield ("vision_tower.vision_model.post_layernorm.weight", np.ascontiguousarray(post_ffw_all[-1]))


def _iter_moe_transformer(model_path: Path, keys: list[str], num_layers: int) -> Iterator[tuple[str, np.ndarray]]:
    """Yield tensors for the Gemma 4 MoE (26B-A4B-it) two-branch architecture.

    Every MoE layer has both a dense MLP (``mlp2.*``) and a routed MoE block
    (``mlp.*``); their outputs sum into the residual stream. This iterator
    emits all per-layer tensors the Mojo MoE forward pass consumes:

    * Attention (via :func:`_iter_attn_layer` — handles sliding vs full layouts)
    * Input/post-attention norms (same names as base)
    * Dense MLP branch: ``mlp.gate_proj`` / ``up_proj`` / ``down_proj``
      (derived from ``mlp2.gating_einsum.w`` + ``mlp2.linear.w``)
    * Router: ``moe_router.proj.weight`` ``[E, H]``, ``moe_router.scale``,
      ``moe_router.per_expert_scale``
    * Experts packed: ``moe_experts.gate_up_proj`` ``[E, 2·I_moe, H]``,
      ``moe_experts.down_proj`` ``[E, H, I_moe]``
    * MoE norms: ``pre_feedforward_layernorm`` (dense pre),
      ``post_feedforward_layernorm_1`` (dense post),
      ``pre_feedforward_layernorm_2`` (MoE pre),
      ``post_feedforward_layernorm_2`` (MoE post), plus the opaque
      ``post_feedforward_layernorm`` (role unclear in the HF reference —
      emitted faithfully when present)
    * ``moe_skip_scale.weight`` — scalar ``(1,)`` per layer, present only
      in Orbax; emitted verbatim so the Mojo forward pass can decide
      whether to apply ``residual * skip_scale`` after inspection.
    """
    embed = OrbaxLoader.open_tensor(model_path, "embedder.input_embedding")
    yield "model.embed_tokens.weight", embed
    yield "lm_head.weight", embed

    yield "model.norm.weight", OrbaxLoader.open_tensor(model_path, "final_norm.scale")

    keys_set = set(keys)

    for n in range(num_layers):
        pfx_in = f"layer_{n}"
        pfx_out = f"model.layers.{n}"

        # Input/post-attention norms (shared with dense layers).
        yield (
            f"{pfx_out}.input_layernorm.weight",
            OrbaxLoader.open_tensor(model_path, f"{pfx_in}.pre_attention_norm.scale"),
        )
        yield (
            f"{pfx_out}.post_attention_layernorm.weight",
            OrbaxLoader.open_tensor(model_path, f"{pfx_in}.post_attention_norm.scale"),
        )

        yield from _iter_attn_layer(model_path, keys_set, n)

        # Dense MLP branch (from Orbax mlp2.*)
        dense_gate_up = OrbaxLoader.open_tensor(model_path, f"{pfx_in}.mlp2.gating_einsum.w")
        yield f"{pfx_out}.mlp.gate_proj.weight", dense_gate_up[0]
        yield f"{pfx_out}.mlp.up_proj.weight", dense_gate_up[1]
        dense_down = OrbaxLoader.open_tensor(model_path, f"{pfx_in}.mlp2.linear.w")
        yield f"{pfx_out}.mlp.down_proj.weight", dense_down.T

        # Router
        router = OrbaxLoader.open_tensor(model_path, f"{pfx_in}.mlp.router_logits.w")
        # Orbax stores [H, E]; Mojo expects [E, H] so the router gemm is
        # (weights @ hidden) producing [E] logits per token — same convention
        # as every other projection in the file.
        yield f"{pfx_out}.moe_router.proj.weight", router.T
        yield (f"{pfx_out}.moe_router.scale", OrbaxLoader.open_tensor(model_path, f"{pfx_in}.mlp.router_scale"))
        yield (
            f"{pfx_out}.moe_router.per_expert_scale",
            OrbaxLoader.open_tensor(model_path, f"{pfx_in}.mlp.per_expert_scale"),
        )

        # Experts (packed gate_up + down)
        expert_gate_up = OrbaxLoader.open_tensor(model_path, f"{pfx_in}.mlp.gating_einsum.w")
        # Orbax [E, 2, I_moe, H] → [E, 2·I_moe, H] preserving gate-then-up
        # interleave on axis 1 (matches HF gate_up_proj layout).
        e, _two, i_moe, h = expert_gate_up.shape
        yield (f"{pfx_out}.moe_experts.gate_up_proj", expert_gate_up.reshape(e, 2 * i_moe, h))
        expert_down = OrbaxLoader.open_tensor(model_path, f"{pfx_in}.mlp.linear.w")
        # Orbax [E, I_moe, H] → [E, H, I_moe] (transpose last two axes).
        yield f"{pfx_out}.moe_experts.down_proj", expert_down.transpose(0, 2, 1)

        # MoE norms — emit every one present in the inventory.
        yield (
            f"{pfx_out}.pre_feedforward_layernorm.weight",
            OrbaxLoader.open_tensor(model_path, f"{pfx_in}.pre_ffw_norm.scale"),
        )
        yield (
            f"{pfx_out}.post_feedforward_layernorm_1.weight",
            OrbaxLoader.open_tensor(model_path, f"{pfx_in}.post_ffw1_norm.scale"),
        )
        yield (
            f"{pfx_out}.pre_feedforward_layernorm_2.weight",
            OrbaxLoader.open_tensor(model_path, f"{pfx_in}.pre_ffw2_norm.scale"),
        )
        yield (
            f"{pfx_out}.post_feedforward_layernorm_2.weight",
            OrbaxLoader.open_tensor(model_path, f"{pfx_in}.post_ffw2_norm.scale"),
        )
        if f"{pfx_in}.post_ffw_norm.scale" in keys_set:
            yield (
                f"{pfx_out}.post_feedforward_layernorm.weight",
                OrbaxLoader.open_tensor(model_path, f"{pfx_in}.post_ffw_norm.scale"),
            )

        # skip_scale scalar — preserved verbatim.
        if f"{pfx_in}.skip_scale" in keys_set:
            yield (f"{pfx_out}.moe_skip_scale.weight", OrbaxLoader.open_tensor(model_path, f"{pfx_in}.skip_scale"))


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


# ── Public entry point ───────────────────────────────────────────────────────


_DEFAULT_SHARD_SIZE_BYTES = 5 * 1024 * 1024 * 1024  # 5 GiB per shard


def convert_orbax_to_safetensors(model_path: Path, *, shard_size_bytes: int = _DEFAULT_SHARD_SIZE_BYTES) -> list[Path]:
    """Convert an Orbax/OCDBT checkpoint at ``model_path`` to HF-style safetensors.

    Reads the checkpoint with streaming :class:`OrbaxLoader` helpers, selects the
    per-variant iterator based on which Orbax keys are present, writes
    ``model.safetensors`` (or sharded equivalents) into ``model_path`` alongside
    the existing Orbax artifacts, and emits ``config.json`` if one isn't already
    present.

    The caller is responsible for cleaning up the original Orbax artifacts after
    this function returns successfully (see
    :meth:`HubManager._cleanup_orbax_artifacts`).

    Args:
        model_path: Directory containing the Orbax checkpoint. Safetensors files
            are written into the same directory.
        shard_size_bytes: Maximum bytes per safetensors shard. A single tensor
            exceeding this still occupies its own shard.

    Returns:
        List of safetensors files written, in shard order.

    Raises:
        ValueError: If the checkpoint has no recognizable variant keys.
        NotImplementedError: If the variant is ``moe`` (MoE support is pending).
    """
    keys = OrbaxLoader.enumerate_tensors(model_path)
    if not keys:
        msg = f"No Orbax tensors found under {model_path}"
        raise ValueError(msg)

    variant = _variant_from_keys(keys)
    num_layers = _layer_count(keys)

    iterators: list[Iterator[tuple[str, np.ndarray]]]
    if variant == "moe":
        iterators = [_iter_moe_transformer(model_path, keys, num_layers)]
    else:
        iterators = [_iter_base_transformer(model_path, keys, num_layers)]
        if variant == "ple":
            iterators.append(_iter_ple(model_path, num_layers))

    if any("vision_encoder." in k for k in keys):
        vision_layer_count = _count_vision_layers(keys, model_path)
        if vision_layer_count > 0:
            iterators.append(_iter_vision(model_path, vision_layer_count))

    def _chain_all() -> Iterator[tuple[str, np.ndarray]]:
        for it in iterators:
            yield from it

    shards = _write_sharded(model_path, _chain_all(), shard_size_bytes)

    config_path = model_path / "config.json"
    if not config_path.exists():

        def _shape_oracle(name: str) -> tuple[int, ...]:
            return OrbaxLoader.open_tensor(model_path, name).shape

        config = _generate_config_json(keys, _shape_oracle)
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    logger.info("Converted %s → %d safetensors shard(s)", model_path, len(shards))
    return shards


_VISION_LAYER_RE = re.compile(r"vision_encoder\.transformer\.stacked_layers\.block\.")


def _count_vision_layers(keys: list[str], model_path: Path) -> int:
    """Infer the vision layer count from the stacked-layer tensors' leading axis.

    Vision tensors are vmapped — their leading axis is the layer index. Open any
    one and return that dimension.
    """
    for key in keys:
        if _VISION_LAYER_RE.search(key):
            shape = OrbaxLoader.open_tensor(model_path, key).shape
            return int(shape[0])
    return 0
