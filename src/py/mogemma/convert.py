"""Convert Orbax/OCDBT checkpoints to HuggingFace-style safetensors.

Google's GCS ``gemma-data`` bucket distributes Gemma 3 weights in Orbax
format (Zarr arrays inside an OCDBT key-value store).  The Mojo backend
expects HuggingFace naming conventions with separate projection matrices
and Float32 data.  This module bridges that gap as a one-time conversion
that runs after the initial GCS download.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from safetensors.numpy import save_file

from .orbax_loader import OrbaxLoader

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

_LAYER_RE = re.compile(r"transformer/layer_(\d+)/(.+)")


def _to_f32(arr: npt.NDArray[np.generic]) -> npt.NDArray[np.float32]:
    """Cast to float32, handling bfloat16 via an intermediate float64 cast."""
    if arr.dtype == np.float32:
        return arr  # type: ignore[return-value]
    return arr.astype(np.float32)


def _convert_gemma3(orbax: dict[str, npt.NDArray[np.generic]]) -> dict[str, npt.NDArray[np.float32]]:
    """Convert a standard Gemma 3 Orbax checkpoint to HuggingFace layout."""
    out: dict[str, npt.NDArray[np.float32]] = {}

    # --- Global tensors ---
    embed = _to_f32(orbax["transformer/embedder.input_embedding"])
    out["model.embed_tokens.weight"] = embed
    out["lm_head.weight"] = embed  # tied weights
    out["model.norm.weight"] = _to_f32(orbax["transformer/final_norm.scale"])

    # --- Per-layer tensors ---
    layer_indices: set[int] = set()
    for name in orbax:
        m = _LAYER_RE.match(name)
        if m:
            layer_indices.add(int(m.group(1)))

    for n in sorted(layer_indices):
        prefix = f"transformer/layer_{n}"
        hf = f"model.layers.{n}"

        # Norms (4 in Orbax → 4 in HF safetensors; Mojo loads the 2 it needs)
        out[f"{hf}.input_layernorm.weight"] = _to_f32(orbax[f"{prefix}/pre_attention_norm.scale"])
        out[f"{hf}.post_attention_layernorm.weight"] = _to_f32(orbax[f"{prefix}/post_attention_norm.scale"])
        out[f"{hf}.pre_feedforward_layernorm.weight"] = _to_f32(orbax[f"{prefix}/pre_ffw_norm.scale"])
        out[f"{hf}.post_feedforward_layernorm.weight"] = _to_f32(orbax[f"{prefix}/post_ffw_norm.scale"])

        # QK Norms
        out[f"{hf}.self_attn.q_norm.weight"] = _to_f32(orbax[f"{prefix}/attn/_query_norm.scale"])
        out[f"{hf}.self_attn.k_norm.weight"] = _to_f32(orbax[f"{prefix}/attn/_key_norm.scale"])

        # Q projection: (num_heads, hidden, head_dim) → (num_heads*head_dim, hidden)
        q_ein = orbax[f"{prefix}/attn/q_einsum.w"]
        out[f"{hf}.self_attn.q_proj.weight"] = _to_f32(q_ein.transpose(0, 2, 1).reshape(-1, q_ein.shape[1]))

        # KV projection: (2, num_kv_heads, hidden, head_dim) → separate k/v
        kv_ein = orbax[f"{prefix}/attn/kv_einsum.w"]
        for idx, name in enumerate(("k_proj", "v_proj")):
            single = kv_ein[idx]  # (num_kv_heads, hidden, head_dim)
            out[f"{hf}.self_attn.{name}.weight"] = _to_f32(single.transpose(0, 2, 1).reshape(-1, single.shape[1]))

        # O projection: (num_heads, head_dim, hidden) → (hidden, num_heads*head_dim)
        o_ein = orbax[f"{prefix}/attn/attn_vec_einsum.w"]
        out[f"{hf}.self_attn.o_proj.weight"] = _to_f32(o_ein.reshape(-1, o_ein.shape[2]).T)

        # Gate / Up: (2, intermediate, hidden) → split
        gating = orbax[f"{prefix}/mlp/gating_einsum.w"]
        out[f"{hf}.mlp.gate_proj.weight"] = _to_f32(gating[0])
        out[f"{hf}.mlp.up_proj.weight"] = _to_f32(gating[1])

        # Down: (intermediate, hidden) → (hidden, intermediate)
        out[f"{hf}.mlp.down_proj.weight"] = _to_f32(orbax[f"{prefix}/mlp/linear.w"].T)

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert_orbax_to_safetensors(model_path: Path) -> Path:
    """Convert an Orbax checkpoint at *model_path* to safetensors in-place.

    Returns the path to the generated ``model.safetensors`` file.
    """
    if not OrbaxLoader.can_load(model_path):
        msg = f"No Orbax checkpoint found at {model_path}"
        raise FileNotFoundError(msg)

    logger.info("Converting Orbax checkpoint at %s to safetensors …", model_path)

    loader = OrbaxLoader(model_path)
    try:
        hf_tensors = _convert_gemma3(loader._arrays)  # noqa: SLF001
    finally:
        loader.close()

    out_path = model_path / "model.safetensors"

    # safetensors.numpy.save_file expects contiguous arrays
    contiguous: dict[str, npt.NDArray[np.float32]] = {k: np.ascontiguousarray(v) for k, v in hf_tensors.items()}
    save_file(contiguous, str(out_path))

    logger.info("Wrote %d tensors to %s", len(contiguous), out_path)
    return out_path
