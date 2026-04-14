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

import logging
import re
from typing import TYPE_CHECKING, Literal

from mogemma.orbax_loader import OrbaxLoader

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import numpy as np

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


def _iter_base_transformer(
    model_path: Path, _keys: list[str], num_layers: int
) -> Iterator[tuple[str, np.ndarray]]:
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
