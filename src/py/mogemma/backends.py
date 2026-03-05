"""Backend contracts and resolution helpers for Python runtime dispatch."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy.typing as npt

TensorMetadata = dict[str, tuple[int, tuple[int, ...], str]]

_CPU_BACKEND_ID = "cpu_mojo"
_GPU_BACKEND_ID = "gpu_mojo"
_BACKEND_ALIASES = {
    "cpu": _CPU_BACKEND_ID,
    _CPU_BACKEND_ID: _CPU_BACKEND_ID,
    "gpu": _GPU_BACKEND_ID,
    _GPU_BACKEND_ID: _GPU_BACKEND_ID,
}


class GenerationBackend(Protocol):
    """Contract for generation backends."""

    backend_id: str

    def init_model(self, metadata: TensorMetadata) -> object:
        """Initialize backend runtime state from tensor metadata."""
        ...

    def step(self, llm: object, token_id: int, temp: float, top_k: int, top_p: float) -> npt.ArrayLike:
        """Run one autoregressive step and return logits."""
        ...


class EmbeddingBackend(Protocol):
    """Contract for embedding backends."""

    backend_id: str

    def init_model(self, metadata: TensorMetadata) -> object:
        """Initialize backend runtime state from tensor metadata."""
        ...

    def generate_embeddings(self, llm: object, tokens: Sequence[Sequence[int]]) -> npt.ArrayLike:
        """Generate embeddings for batched token rows."""
        ...


class CPUCoreBackend:
    """Adapter around the compiled `mogemma._core` module."""

    backend_id = _CPU_BACKEND_ID

    def __init__(self, core_module: object) -> None:
        self._core = core_module

    def init_model(self, metadata: TensorMetadata) -> object:
        return self._core.init_model(metadata)

    def step(self, llm: object, token_id: int, temp: float, top_k: int, top_p: float) -> npt.ArrayLike:
        return self._core.step(llm, token_id, temp, top_k, top_p)

    def generate_embeddings(self, llm: object, tokens: Sequence[Sequence[int]]) -> npt.ArrayLike:
        return self._core.generate_embeddings(llm, tokens)


def resolve_backend_id(device: str) -> str:
    """Resolve user-facing device/backend values into canonical backend IDs."""
    normalized = device.strip().lower()
    if not normalized:
        return _CPU_BACKEND_ID

    backend_id = _BACKEND_ALIASES.get(normalized)
    if backend_id is None:
        supported = ", ".join(sorted(_BACKEND_ALIASES))
        msg = f"Unsupported backend '{device}'. Supported backends: {supported}"
        raise ValueError(msg)
    return backend_id


def resolve_generation_backend(*, device: str, core_module: object) -> GenerationBackend:
    """Resolve the active generation backend implementation."""
    backend_id = resolve_backend_id(device)
    if backend_id != _CPU_BACKEND_ID:
        msg = (
            f"Unsupported backend '{device}' (resolved as '{backend_id}'). "
            "Currently available runtime backend: cpu, cpu_mojo"
        )
        raise ValueError(msg)
    return CPUCoreBackend(core_module)


def resolve_embedding_backend(*, device: str, core_module: object) -> EmbeddingBackend:
    """Resolve the active embedding backend implementation."""
    backend_id = resolve_backend_id(device)
    if backend_id != _CPU_BACKEND_ID:
        msg = (
            f"Unsupported backend '{device}' (resolved as '{backend_id}'). "
            "Currently available runtime backend: cpu, cpu_mojo"
        )
        raise ValueError(msg)
    return CPUCoreBackend(core_module)
