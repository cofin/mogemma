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
