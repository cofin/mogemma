"""Backend contracts and resolution helpers for Python runtime dispatch."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class DeviceSelection:
    """Normalized device-selection result used by model init paths."""

    requested: str
    effective_backend_id: str
    effective_device: str
    gpu_index: int | None
    used_fallback: bool


class CoreModuleContract(Protocol):
    """Required callable surface exposed by `mogemma._core`."""

    def init_model(self, metadata: TensorMetadata) -> object:
        """Initialize a backend model session from tensor metadata."""
        ...

    def step(self, llm: object, token_id: int, temp: float, top_k: int, top_p: float) -> npt.ArrayLike:
        """Run one token step and return logits."""
        ...

    def generate_embeddings(self, llm: object, tokens: Sequence[Sequence[int]]) -> npt.ArrayLike:
        """Run batched embedding inference and return embedding rows."""
        ...


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

    def __init__(self, core_module: CoreModuleContract) -> None:
        self._core = core_module

    def init_model(self, metadata: TensorMetadata) -> object:
        return self._core.init_model(metadata)

    def step(self, llm: object, token_id: int, temp: float, top_k: int, top_p: float) -> npt.ArrayLike:
        return self._core.step(llm, token_id, temp, top_k, top_p)

    def generate_embeddings(self, llm: object, tokens: Sequence[Sequence[int]]) -> npt.ArrayLike:
        return self._core.generate_embeddings(llm, tokens)


def resolve_backend_id(device: str) -> str:
    """Resolve user-facing device/backend values into canonical backend IDs.

    Resolution order:
    1. Empty/whitespace -> `cpu_mojo`.
    2. Exact canonical IDs and known aliases (`cpu`, `cpu_mojo`, `gpu`, `gpu_mojo`).
    3. GPU selectors of the form `gpu:N` (future-compatible with device index).
    """
    normalized = device.strip().lower()
    if not normalized:
        return _CPU_BACKEND_ID

    backend_id = _BACKEND_ALIASES.get(normalized)
    if backend_id is not None:
        return backend_id

    if normalized.startswith("gpu:"):
        _, _, selector = normalized.partition(":")
        if selector.isdigit():
            return _GPU_BACKEND_ID

    supported = "cpu, cpu_mojo, gpu, gpu_mojo, gpu:<index>"
    msg = f"Unsupported backend '{device}'. Supported backends: {supported}"
    raise ValueError(msg)


def parse_device_spec(device: str) -> tuple[str, int | None]:
    """Parse a device spec into `(backend_id, gpu_index)`."""
    normalized = device.strip().lower()
    if not normalized:
        normalized = "cpu"

    backend_id = resolve_backend_id(normalized)
    gpu_index: int | None = None
    if normalized.startswith("gpu:"):
        _, _, selector = normalized.partition(":")
        gpu_index = int(selector)
    return backend_id, gpu_index


def gpu_capability_available() -> bool:
    """Best-effort GPU capability hook.

    Current policy hook is env-based to keep behavior deterministic in tests and
    CPU-only hosts. Runtime integration can replace this with a richer probe.
    """
    value = os.getenv("MOGEMMA_GPU_AVAILABLE", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def resolve_device_selection(
    device: str,
    *,
    allow_fallback: bool,
    gpu_available: bool | None = None,
) -> DeviceSelection:
    """Resolve requested device into a concrete backend selection policy."""
    backend_id, gpu_index = parse_device_spec(device)
    if backend_id == _CPU_BACKEND_ID:
        return DeviceSelection(
            requested=device,
            effective_backend_id=_CPU_BACKEND_ID,
            effective_device="cpu",
            gpu_index=None,
            used_fallback=False,
        )

    available = gpu_capability_available() if gpu_available is None else gpu_available
    requested_label = "gpu" if gpu_index is None else f"gpu:{gpu_index}"
    if available:
        return DeviceSelection(
            requested=device,
            effective_backend_id=_GPU_BACKEND_ID,
            effective_device=requested_label,
            gpu_index=gpu_index,
            used_fallback=False,
        )

    if allow_fallback:
        return DeviceSelection(
            requested=device,
            effective_backend_id=_CPU_BACKEND_ID,
            effective_device="cpu",
            gpu_index=None,
            used_fallback=True,
        )

    msg = (
        f"Requested device '{requested_label}' is unavailable. "
        "Set allow_device_fallback=True to permit deterministic fallback to cpu."
    )
    raise RuntimeError(msg)


def resolve_generation_backend(*, device: str, core_module: object) -> GenerationBackend:
    """Resolve the active generation backend implementation."""
    backend_id = resolve_backend_id(device)
    if backend_id != _CPU_BACKEND_ID:
        msg = (
            f"Unsupported backend '{device}' (resolved as '{backend_id}'). "
            "Currently available runtime backend: cpu, cpu_mojo"
        )
        raise ValueError(msg)
    _validate_core_module(core_module, required=("init_model", "step"))
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
    _validate_core_module(core_module, required=("init_model", "generate_embeddings"))
    return CPUCoreBackend(core_module)


def _validate_core_module(core_module: object, *, required: tuple[str, ...]) -> None:
    missing = [name for name in required if not callable(getattr(core_module, name, None))]
    if missing:
        msg = (
            "Invalid core module for cpu_mojo backend: missing callables "
            + ", ".join(missing)
            + "."
        )
        raise TypeError(msg)
