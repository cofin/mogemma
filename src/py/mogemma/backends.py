"""Backend contracts and resolution helpers for Python runtime dispatch."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

TensorMetadata = dict[str, tuple[int, tuple[int, ...], str]]

_CPU_BACKEND_ID = "cpu"
_GPU_BACKEND_ID = "gpu"


@dataclass(frozen=True)
class DeviceSelection:
    """Normalized device-selection result used by model init paths."""

    requested: str
    backend: str
    device_kind: str
    device_index: int | None
    strict: bool
    availability_source: str

    @property
    def effective_backend_id(self) -> str:
        """Get the resolved backend identifier."""
        return self.backend

    @property
    def effective_device(self) -> str:
        """Get the effective device string."""
        if self.device_kind == _CPU_BACKEND_ID:
            return _CPU_BACKEND_ID
        if self.device_index is not None:
            return f"{self.device_kind}:{self.device_index}"
        return self.device_kind

    @property
    def gpu_index(self) -> int | None:
        """Get the parsed GPU index, if any."""
        return self.device_index

    def as_runtime_descriptor(self) -> dict[str, object]:
        """Convert selection to a dictionary for Mojo core initialization."""
        return {
            "requested": self.requested,
            "backend": self.backend,
            "device_kind": self.device_kind,
            "device_index": self.device_index,
            "strict": self.strict,
            "availability_source": self.availability_source,
        }


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
    """Wrapper mapping to the standard generic CPU backend implementation."""

    backend_id = _CPU_BACKEND_ID

    def __init__(self, core_module: CoreModuleContract) -> None:
        """Initialize the CPU core backend wrapper."""
        self._core = core_module

    def init_model(self, metadata: TensorMetadata) -> object:
        """Initialize the model context inside the core backend."""
        return self._core.init_model(metadata)

    def step(self, llm: object, token_id: int, temp: float, top_k: int, top_p: float) -> npt.ArrayLike:
        """Step the LLM context to decode the next token logits."""
        return self._core.step(llm, token_id, temp, top_k, top_p)

    def generate_embeddings(self, llm: object, tokens: Sequence[Sequence[int]]) -> npt.ArrayLike:
        """Generate numerical embeddings from sequences of tokens."""
        return self._core.generate_embeddings(llm, tokens)


def resolve_backend_id(device: str) -> str:
    """Resolve user-facing device/backend values into canonical backend IDs.

    Resolution order:
    1. Empty/whitespace -> `cpu`.
    2. Exact canonical IDs (`cpu`, `gpu`).
    3. GPU selectors of the form `gpu:N`.
    """
    normalized = device.strip().lower()
    if not normalized:
        return _CPU_BACKEND_ID

    if normalized in (_CPU_BACKEND_ID, _GPU_BACKEND_ID):
        return normalized

    if normalized.startswith("gpu:"):
        _, _, selector = normalized.partition(":")
        if selector.isdigit():
            return _GPU_BACKEND_ID

    supported = "cpu, gpu, gpu:<index>"
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


def resolve_device_selection(device: str, *, gpu_available: bool | None = None) -> DeviceSelection:
    """Resolve requested device into a concrete backend selection."""
    normalized = device.strip().lower() or "cpu"
    backend_id, gpu_index = parse_device_spec(device)
    if backend_id == _CPU_BACKEND_ID:
        return DeviceSelection(
            requested=normalized,
            backend=_CPU_BACKEND_ID,
            device_kind=_CPU_BACKEND_ID,
            device_index=None,
            strict=False,
            availability_source="cpu-default",
        )

    available = gpu_capability_available() if gpu_available is None else gpu_available
    requested_label = "gpu" if gpu_index is None else f"gpu:{gpu_index}"
    if available:
        return DeviceSelection(
            requested=normalized,
            backend=_GPU_BACKEND_ID,
            device_kind=_GPU_BACKEND_ID,
            device_index=gpu_index,
            strict=True,
            availability_source="override" if gpu_available is not None else "env:MOGEMMA_GPU_AVAILABLE",
        )
    msg = f"Requested device '{requested_label}' is unavailable on this host."
    raise RuntimeError(msg)


def resolve_generation_backend(*, device: str, core_module: object) -> GenerationBackend:
    """Resolve the active generation backend implementation."""
    backend_id = resolve_backend_id(device)
    if backend_id != _CPU_BACKEND_ID:
        msg = f"Unsupported backend '{device}' (resolved as '{backend_id}'). Currently available runtime backend: cpu"
        raise ValueError(msg)
    _validate_core_module(core_module, required=("init_model", "step"))
    return CPUCoreBackend(cast("CoreModuleContract", core_module))


def resolve_embedding_backend(*, device: str, core_module: object) -> EmbeddingBackend:
    """Resolve the active embedding backend implementation."""
    backend_id = resolve_backend_id(device)
    if backend_id != _CPU_BACKEND_ID:
        msg = f"Unsupported backend '{device}' (resolved as '{backend_id}'). Currently available runtime backend: cpu"
        raise ValueError(msg)
    _validate_core_module(core_module, required=("init_model", "generate_embeddings"))
    return CPUCoreBackend(cast("CoreModuleContract", core_module))


def _validate_core_module(core_module: object, *, required: tuple[str, ...]) -> None:
    missing = [name for name in required if not callable(getattr(core_module, name, None))]
    if missing:
        msg = "Invalid core module for cpu backend: missing callables " + ", ".join(missing) + "."
        raise TypeError(msg)
