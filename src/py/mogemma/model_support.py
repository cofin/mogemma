"""Gemma 4 official model and runtime support metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .hub import KNOWN_GCS_MODELS


class RuntimeSupport(str, Enum):
    """Runtime support state for a model capability."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    REQUIRES_FOLLOWUP = "requires_followup"


@dataclass(frozen=True)
class Gemma4ModelSupport:
    """Support metadata for one official Gemma 4 model id."""

    model_id: str
    variant: str
    official: bool
    gcs_available: bool
    text: RuntimeSupport
    image: RuntimeSupport
    audio: RuntimeSupport
    moe: RuntimeSupport
    ple: RuntimeSupport
    mtp: RuntimeSupport
    thinking: RuntimeSupport
    unified_multimodal: RuntimeSupport
    notes: str


OFFICIAL_GEMMA4_MODELS: frozenset[str] = frozenset({
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "google/gemma-4-12B-it",
    "google/gemma-4-31B-it",
    "google/gemma-4-26B-A4B-it",
})
"""Official Gemma 4 model ids known as of 2026-07-05."""


GEMMA4_MODEL_SUPPORT: dict[str, Gemma4ModelSupport] = {
    "google/gemma-4-E2B-it": Gemma4ModelSupport(
        model_id="google/gemma-4-E2B-it",
        variant="gemma4_dense_e2b",
        official=True,
        gcs_available="google/gemma-4-E2B-it" in KNOWN_GCS_MODELS,
        text=RuntimeSupport.SUPPORTED,
        image=RuntimeSupport.SUPPORTED,
        audio=RuntimeSupport.UNSUPPORTED,
        moe=RuntimeSupport.UNSUPPORTED,
        ple=RuntimeSupport.SUPPORTED,
        mtp=RuntimeSupport.REQUIRES_FOLLOWUP,
        thinking=RuntimeSupport.REQUIRES_FOLLOWUP,
        unified_multimodal=RuntimeSupport.UNSUPPORTED,
        notes="PLE text/image runtime is present; audio, MTP, and thinking templates are not implemented.",
    ),
    "google/gemma-4-E4B-it": Gemma4ModelSupport(
        model_id="google/gemma-4-E4B-it",
        variant="gemma4_dense_e4b",
        official=True,
        gcs_available="google/gemma-4-E4B-it" in KNOWN_GCS_MODELS,
        text=RuntimeSupport.SUPPORTED,
        image=RuntimeSupport.SUPPORTED,
        audio=RuntimeSupport.UNSUPPORTED,
        moe=RuntimeSupport.UNSUPPORTED,
        ple=RuntimeSupport.SUPPORTED,
        mtp=RuntimeSupport.REQUIRES_FOLLOWUP,
        thinking=RuntimeSupport.REQUIRES_FOLLOWUP,
        unified_multimodal=RuntimeSupport.UNSUPPORTED,
        notes="PLE text/image runtime is present; audio, MTP, and thinking templates are not implemented.",
    ),
    "google/gemma-4-12B-it": Gemma4ModelSupport(
        model_id="google/gemma-4-12B-it",
        variant="gemma4_dense_12b_unified",
        official=True,
        gcs_available="google/gemma-4-12B-it" in KNOWN_GCS_MODELS,
        text=RuntimeSupport.UNSUPPORTED,
        image=RuntimeSupport.UNSUPPORTED,
        audio=RuntimeSupport.UNSUPPORTED,
        moe=RuntimeSupport.UNSUPPORTED,
        ple=RuntimeSupport.UNSUPPORTED,
        mtp=RuntimeSupport.REQUIRES_FOLLOWUP,
        thinking=RuntimeSupport.REQUIRES_FOLLOWUP,
        unified_multimodal=RuntimeSupport.REQUIRES_FOLLOWUP,
        notes=(
            "Official unified encoder-free multimodal architecture; recognized but runtime support is not implemented."
        ),
    ),
    "google/gemma-4-31B-it": Gemma4ModelSupport(
        model_id="google/gemma-4-31B-it",
        variant="gemma4_dense_31b",
        official=True,
        gcs_available="google/gemma-4-31B-it" in KNOWN_GCS_MODELS,
        text=RuntimeSupport.REQUIRES_FOLLOWUP,
        image=RuntimeSupport.REQUIRES_FOLLOWUP,
        audio=RuntimeSupport.UNSUPPORTED,
        moe=RuntimeSupport.UNSUPPORTED,
        ple=RuntimeSupport.UNSUPPORTED,
        mtp=RuntimeSupport.REQUIRES_FOLLOWUP,
        thinking=RuntimeSupport.REQUIRES_FOLLOWUP,
        unified_multimodal=RuntimeSupport.UNSUPPORTED,
        notes="Official dense model; no checked-in GCS availability or live parity evidence in this repo.",
    ),
    "google/gemma-4-26B-A4B-it": Gemma4ModelSupport(
        model_id="google/gemma-4-26B-A4B-it",
        variant="gemma4_moe_26b",
        official=True,
        gcs_available="google/gemma-4-26B-A4B-it" in KNOWN_GCS_MODELS,
        text=RuntimeSupport.REQUIRES_FOLLOWUP,
        image=RuntimeSupport.REQUIRES_FOLLOWUP,
        audio=RuntimeSupport.UNSUPPORTED,
        moe=RuntimeSupport.REQUIRES_FOLLOWUP,
        ple=RuntimeSupport.UNSUPPORTED,
        mtp=RuntimeSupport.REQUIRES_FOLLOWUP,
        thinking=RuntimeSupport.REQUIRES_FOLLOWUP,
        unified_multimodal=RuntimeSupport.UNSUPPORTED,
        notes="MoE runtime path exists, but live parity and end-to-end support remain gated by follow-up validation.",
    ),
}


def get_gemma4_model_support(model_id: str) -> Gemma4ModelSupport:
    """Return support metadata for an official Gemma 4 model id."""
    try:
        return GEMMA4_MODEL_SUPPORT[model_id]
    except KeyError as exc:
        msg = f"Unknown official Gemma 4 model id: {model_id}"
        raise KeyError(msg) from exc


__all__ = [
    "GEMMA4_MODEL_SUPPORT",
    "OFFICIAL_GEMMA4_MODELS",
    "Gemma4ModelSupport",
    "RuntimeSupport",
    "get_gemma4_model_support",
]
