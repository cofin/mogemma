from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from mogemma.backends import CPUCoreBackend, resolve_backend_id, resolve_embedding_backend, resolve_generation_backend


def test_resolve_backend_id_defaults_to_cpu_mojo() -> None:
    assert resolve_backend_id("") == "cpu_mojo"
    assert resolve_backend_id("   ") == "cpu_mojo"


def test_resolve_backend_id_normalizes_known_aliases() -> None:
    assert resolve_backend_id("cpu") == "cpu_mojo"
    assert resolve_backend_id("CPU_MOJO") == "cpu_mojo"
    assert resolve_backend_id("gpu") == "gpu_mojo"
    assert resolve_backend_id("GPU_MOJO") == "gpu_mojo"


def test_resolve_backend_id_accepts_gpu_device_selector() -> None:
    assert resolve_backend_id("gpu:0") == "gpu_mojo"
    assert resolve_backend_id("gpu:17") == "gpu_mojo"


def test_resolve_backend_id_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="Unsupported backend"):
        resolve_backend_id("banana")


class _CoreSpy:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def init_model(self, metadata: object) -> object:
        self.calls.append(("init_model", metadata))
        return {"llm": "state"}

    def step(self, llm: object, token_id: int, temp: float, top_k: int, top_p: float) -> npt.NDArray[np.float32]:
        self.calls.append(("step", (llm, token_id, temp, top_k, top_p)))
        return np.array([1.0, 0.0], dtype=np.float32)

    def generate_embeddings(self, llm: object, tokens: list[list[int]]) -> npt.NDArray[np.float32]:
        self.calls.append(("generate_embeddings", (llm, tokens)))
        return np.ones((len(tokens), 4), dtype=np.float32)


def test_resolve_generation_backend_uses_cpu_core_adapter() -> None:
    core = _CoreSpy()

    backend = resolve_generation_backend(device="cpu", core_module=core)

    assert isinstance(backend, CPUCoreBackend)
    llm = backend.init_model({"weight": (1, (2, 3), "f32")})
    logits = backend.step(llm, 7, 0.0, 1, 1.0)

    assert logits.shape == (2,)
    assert core.calls[0][0] == "init_model"
    assert core.calls[1][0] == "step"


def test_resolve_embedding_backend_uses_cpu_core_adapter() -> None:
    core = _CoreSpy()

    backend = resolve_embedding_backend(device="cpu_mojo", core_module=core)

    assert isinstance(backend, CPUCoreBackend)
    llm = backend.init_model({"weight": (1, (2, 3), "f32")})
    embeddings = backend.generate_embeddings(llm, [[1, 2, 3]])

    assert embeddings.shape == (1, 4)
    assert core.calls[0][0] == "init_model"
    assert core.calls[1][0] == "generate_embeddings"


def test_cpu_core_backend_rejects_missing_core_entrypoints() -> None:
    class IncompleteCore:
        def init_model(self, metadata: object) -> object:
            del metadata
            return object()

    with pytest.raises(TypeError, match="missing callables"):
        resolve_generation_backend(device="cpu", core_module=IncompleteCore())
