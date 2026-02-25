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
from collections.abc import Callable, Iterator, Mapping
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
_MIN_LAYER_PARTS = 4
_LAYER_INDEX_PART = 2
_RANK_2D = 2
_RANK_3D = 3
_ALTUP_PROJECTION_COUNT = 3


class _OrbaxLookup(Mapping[str, npt.NDArray[np.generic]]):
    """Read-only mapping adapter that resolves Orbax keys lazily."""

    def __init__(self, getter: Callable[[str], npt.NDArray[np.generic]]) -> None:
        self._getter = getter

    def __getitem__(self, key: str) -> npt.NDArray[np.generic]:
        return self._getter(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0


def _to_f32(arr: npt.NDArray[np.generic]) -> npt.NDArray[np.float32]:
    """Cast to float32, handling bfloat16 via an intermediate float64 cast."""
    if arr.dtype == np.float32:
        return arr  # type: ignore[return-value]
    return arr.astype(np.float32)


def _detect_model_family(orbax_keys: list[str]) -> str:
    """Detect if the checkpoint is standard Gemma 3 or Gemma 3 Nano."""
    for key in orbax_keys:
        if "altup" in key or "per_layer_mapping" in key:
            return "gemma3_nano"
    return "gemma3"


def _convert_norms(
    orbax: Mapping[str, npt.NDArray[np.generic]], out: dict[str, npt.NDArray[np.float32]], prefix: str, hf: str
) -> None:
    out[f"{hf}.input_layernorm.weight"] = _to_f32(orbax[f"{prefix}/pre_attention_norm.scale"])
    out[f"{hf}.post_attention_layernorm.weight"] = _to_f32(orbax[f"{prefix}/post_attention_norm.scale"])
    out[f"{hf}.pre_feedforward_layernorm.weight"] = _to_f32(orbax[f"{prefix}/pre_ffw_norm.scale"])
    out[f"{hf}.post_feedforward_layernorm.weight"] = _to_f32(orbax[f"{prefix}/post_ffw_norm.scale"])


def _convert_attention_layer(  # noqa: PLR0913
    orbax: Mapping[str, npt.NDArray[np.generic]],
    out: dict[str, npt.NDArray[np.float32]],
    prefix: str,
    hf: str,
    q_norm_key: str,
    k_norm_key: str,
) -> None:
    # QK Norms
    out[f"{hf}.self_attn.q_norm.weight"] = _to_f32(orbax[f"{prefix}/attn/{q_norm_key}.scale"])
    out[f"{hf}.self_attn.k_norm.weight"] = _to_f32(orbax[f"{prefix}/attn/{k_norm_key}.scale"])

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


def _convert_mlp_layer(  # noqa: PLR0913
    orbax: Mapping[str, npt.NDArray[np.generic]],
    out: dict[str, npt.NDArray[np.float32]],
    prefix: str,
    hf: str,
    gating_key: str,
    linear_key: str,
) -> None:
    # Gate / Up: (2, intermediate, hidden) → split
    gating = orbax[f"{prefix}/mlp/{gating_key}"]
    out[f"{hf}.mlp.gate_proj.weight"] = _to_f32(gating[0])
    out[f"{hf}.mlp.up_proj.weight"] = _to_f32(gating[1])

    # Down: (intermediate, hidden) → (hidden, intermediate)
    out[f"{hf}.mlp.down_proj.weight"] = _to_f32(orbax[f"{prefix}/mlp/{linear_key}"].T)


def _convert_gemma3(orbax: Mapping[str, npt.NDArray[np.generic]]) -> dict[str, npt.NDArray[np.float32]]:
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

        _convert_norms(orbax, out, prefix, hf)
        _convert_attention_layer(orbax, out, prefix, hf, "_query_norm", "_key_norm")
        _convert_mlp_layer(orbax, out, prefix, hf, "gating_einsum.w", "linear.w")

    return out


def _convert_gemma3_nano(  # noqa: C901, PLR0915
    orbax: Mapping[str, npt.NDArray[np.generic]],
) -> dict[str, npt.NDArray[np.float32]]:
    """Convert a Gemma 3 Nano Orbax checkpoint to HuggingFace layout."""
    out: dict[str, npt.NDArray[np.float32]] = {}

    def get_orbax(key: str) -> npt.NDArray[np.generic]:
        if key in orbax:
            return orbax[key]
        no_prefix = key.replace("transformer/", "")
        if no_prefix in orbax:
            return orbax[no_prefix]
        dot_sep = no_prefix.replace("/", ".")
        if dot_sep in orbax:
            return orbax[dot_sep]
        msg = f"Key not found in Orbax checkpoint: {key}"
        raise KeyError(msg)

    # --- Global tensors ---
    embed = _to_f32(get_orbax("transformer/embedder.input_embedding"))
    out["model.embed_tokens.weight"] = embed
    out["lm_head.weight"] = embed  # tied weights
    out["model.norm.weight"] = _to_f32(get_orbax("transformer/final_norm.scale"))

    out["model.per_layer_embed.weight"] = _to_f32(get_orbax("transformer/embedder.per_layer_input_embedding"))
    out["model.per_layer_embed.projection.weight"] = _to_f32(
        get_orbax("transformer/embedder.per_layer_input_projection.w")
    )
    out["model.per_layer_embed.norm.weight"] = _to_f32(
        get_orbax("transformer/embedder.per_layer_projection_norm.scale")
    )

    # AltUp projections (0, 1, 2)
    for i in range(_ALTUP_PROJECTION_COUNT):
        key_proj = f"altup_projection_{i}.altup_projection_{i}"
        key_unembed = f"altup_unembed_projection_{i}.altup_unembed_projection_{i}"

        if key_proj in orbax:
            out[f"model.altup.projection.{i}.weight"] = _to_f32(orbax[key_proj].T)
            out[f"model.altup.unembed.{i}.weight"] = _to_f32(orbax[key_unembed].T)
        elif f"transformer/altup_projection.{i}" in orbax:
            out[f"model.altup.projection.{i}.weight"] = _to_f32(orbax[f"transformer/altup_projection.{i}"].T)
            out[f"model.altup.unembed.{i}.weight"] = _to_f32(orbax[f"transformer/altup_unembed_projection.{i}"].T)

    # --- Per-layer tensors ---
    layer_indices: set[int] = set()
    layer_re_fallback = re.compile(r"layer_(\d+)[\./](.+)")
    for name in orbax:
        m = _LAYER_RE.match(name)
        if m:
            layer_indices.add(int(m.group(1)))
        else:
            m2 = layer_re_fallback.match(name)
            if m2:
                layer_indices.add(int(m2.group(1)))

    for n in sorted(layer_indices):
        prefix = f"transformer/layer_{n}"
        if f"layer_{n}/pre_attention_norm.scale" in orbax:
            prefix = f"layer_{n}"

        hf = f"model.layers.{n}"

        wrapper = _OrbaxLookup(get_orbax)
        _convert_norms(wrapper, out, prefix, hf)
        _convert_attention_layer(wrapper, out, prefix, hf, "query_norm", "key_norm")
        _convert_mlp_layer(wrapper, out, prefix, hf, "gating_einsum", "linear")

        # Nano-specific: AltUp
        out[f"{hf}.altup.router.weight"] = _to_f32(get_orbax(f"{prefix}/altup.modality_router.w").T)
        out[f"{hf}.altup.router_norm.weight"] = _to_f32(get_orbax(f"{prefix}/altup.router_norm_layer.scale"))
        out[f"{hf}.altup.prediction_coefs"] = _to_f32(get_orbax(f"{prefix}/altup.prediction_coefs"))
        out[f"{hf}.altup.correction_coefs"] = _to_f32(get_orbax(f"{prefix}/altup.correction_coefs"))
        out[f"{hf}.altup.output_scale"] = _to_f32(get_orbax(f"{prefix}/altup.correct_output_scale"))

        # Nano-specific: Per-layer mapping
        out[f"{hf}.per_layer_map.gate.weight"] = _to_f32(
            get_orbax(f"{prefix}/per_layer_mapping.per_layer_input_gate.w").T
        )
        out[f"{hf}.per_layer_map.projection.weight"] = _to_f32(
            get_orbax(f"{prefix}/per_layer_mapping.per_layer_projection.w").T
        )
        out[f"{hf}.per_layer_map.norm.weight"] = _to_f32(
            get_orbax(f"{prefix}/per_layer_mapping.post_per_layer_input_norm.scale")
        )

        # Nano-specific: Laurel
        out[f"{hf}.laurel.down_proj.weight"] = _to_f32(get_orbax(f"{prefix}/laurel.linear_left.w").T)
        out[f"{hf}.laurel.up_proj.weight"] = _to_f32(get_orbax(f"{prefix}/laurel.linear_right.w").T)
        out[f"{hf}.laurel.norm.weight"] = _to_f32(get_orbax(f"{prefix}/post_laurel_norm.scale"))

    _validate_nano_layout(out)
    return out


def _nano_layer_indices(tensors: dict[str, npt.NDArray[np.float32]]) -> list[int]:
    layer_indices: set[int] = set()
    for name in tensors:
        if not name.startswith("model.layers."):
            continue
        parts = name.split(".")
        if len(parts) < _MIN_LAYER_PARTS:
            continue
        try:
            layer_indices.add(int(parts[_LAYER_INDEX_PART]))
        except ValueError:
            continue
    return sorted(layer_indices)


def _validate_nano_layout(  # noqa: C901
    tensors: dict[str, npt.NDArray[np.float32]],
) -> None:
    """Validate tensor-layout assumptions consumed by the Mojo nano path."""
    layer_indices = _nano_layer_indices(tensors)
    if not layer_indices:
        msg = "Converted Nano checkpoint contains no decoder layers"
        raise ValueError(msg)

    hidden_size = int(tensors["model.embed_tokens.weight"].shape[1])
    first_layer = layer_indices[0]
    prefix = f"model.layers.{first_layer}"

    per_layer_embed = tensors["model.per_layer_embed.weight"]
    if per_layer_embed.ndim != _RANK_3D:
        msg = f"Expected model.per_layer_embed.weight rank 3, got rank {per_layer_embed.ndim}"
        raise ValueError(msg)

    plm_gate = tensors[f"{prefix}.per_layer_map.gate.weight"]
    plm_proj = tensors[f"{prefix}.per_layer_map.projection.weight"]
    if plm_gate.ndim != _RANK_2D or plm_proj.ndim != _RANK_2D:
        msg = "Per-layer mapping tensors must be 2D"
        raise ValueError(msg)

    per_layer_dim = int(plm_gate.shape[0])
    if plm_gate.shape[1] != hidden_size:
        msg = f"Per-layer gate hidden_size mismatch: expected second dim {hidden_size}, got {plm_gate.shape[1]}"
        raise ValueError(msg)
    if plm_proj.shape != (hidden_size, per_layer_dim):
        msg = f"Per-layer projection shape mismatch: expected ({hidden_size}, {per_layer_dim}), got {plm_proj.shape}"
        raise ValueError(msg)

    _, embed_layers, embed_dim = per_layer_embed.shape
    if embed_dim != per_layer_dim:
        msg = (
            "Per-layer embedding dim mismatch: expected last dim to match per_layer_dim "
            f"{per_layer_dim}, got {embed_dim}"
        )
        raise ValueError(msg)

    if embed_layers <= max(layer_indices):
        msg = (
            "Per-layer embedding layer axis is smaller than required decoder depth: "
            f"embed_layers={embed_layers}, max_layer_index={max(layer_indices)}"
        )
        raise ValueError(msg)

    router = tensors[f"{prefix}.altup.router.weight"]
    prediction = tensors[f"{prefix}.altup.prediction_coefs"]
    correction = tensors[f"{prefix}.altup.correction_coefs"]
    output_scale = tensors[f"{prefix}.altup.output_scale"]

    if router.ndim != _RANK_2D or router.shape[1] != hidden_size:
        msg = f"AltUp router must be 2D with hidden dim {hidden_size}, got {router.shape}"
        raise ValueError(msg)
    altup_inputs = int(router.shape[0])

    valid_prediction_shape = prediction.shape in {
        (altup_inputs, altup_inputs, altup_inputs),
        (altup_inputs, altup_inputs * altup_inputs),
    }
    if not valid_prediction_shape:
        msg = (
            "AltUp prediction_coefs shape mismatch: expected "
            f"({altup_inputs}, {altup_inputs}, {altup_inputs}) or "
            f"({altup_inputs}, {altup_inputs * altup_inputs}), got {prediction.shape}"
        )
        raise ValueError(msg)
    if correction.shape != (altup_inputs, altup_inputs):
        msg = (
            f"AltUp correction_coefs shape mismatch: expected ({altup_inputs}, {altup_inputs}), got {correction.shape}"
        )
        raise ValueError(msg)
    if output_scale.shape != (hidden_size,):
        msg = f"AltUp output_scale shape mismatch: expected ({hidden_size},), got {output_scale.shape}"
        raise ValueError(msg)


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
        keys = list(loader._arrays.keys())  # noqa: SLF001
        family = _detect_model_family(keys)
        logger.info("Detected model family: %s", family)

        if family == "gemma3_nano":
            hf_tensors = _convert_gemma3_nano(loader._arrays)  # noqa: SLF001
        else:
            hf_tensors = _convert_gemma3(loader._arrays)  # noqa: SLF001
    finally:
        loader.close()

    out_path = model_path / "model.safetensors"

    # safetensors.numpy.save_file expects contiguous arrays
    contiguous: dict[str, npt.NDArray[np.float32]] = {k: np.ascontiguousarray(v) for k, v in hf_tensors.items()}
    save_file(contiguous, str(out_path))

    logger.info("Wrote %d tensors to %s", len(contiguous), out_path)
    return out_path
