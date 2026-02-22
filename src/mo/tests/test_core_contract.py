from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

_core = pytest.importorskip("mogemma._core")


def _install_fake_max_modules(
    monkeypatch: pytest.MonkeyPatch,
    llm_cls: type | None = None,
) -> None:
    """Install lightweight fake `max` Python modules used by core init."""

    class FakePipelineConfig:
        def __init__(self, model_path: str) -> None:
            self.model_path = model_path

    class FakeLLM:
        def __init__(self, pipeline_config: object) -> None:
            self.pipeline_config = pipeline_config

        def step(self, token_id: int, temp: float, top_k: int, top_p: float) -> npt.NDArray[np.float32]:
            del temp, top_k, top_p
            return np.asarray(
                [
                    0.1 + 0.0001 * token_id,
                    0.2 + 0.0002 * token_id,
                    0.3 + 0.0003 * token_id,
                ],
                dtype=np.float32,
            )

        def encode(self, input_array: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
            batch_size = int(input_array.shape[0])
            base = float(np.sum(input_array))
            return np.full((batch_size, 768), base, dtype=np.float32)

    max_module = types.ModuleType("max")
    max_module.__path__ = []  # type: ignore[attr-defined]

    entrypoints_module = types.ModuleType("max.entrypoints")
    entrypoints_module.__path__ = []  # type: ignore[attr-defined]

    entrypoints_llm_module = types.ModuleType("max.entrypoints.llm")
    if llm_cls is None:
        llm_cls = FakeLLM
    entrypoints_llm_module.LLM = llm_cls
    entrypoints_module.llm = entrypoints_llm_module

    pipelines_module = types.ModuleType("max.pipelines")
    pipelines_module.PipelineConfig = FakePipelineConfig

    max_module.entrypoints = entrypoints_module

    monkeypatch.setitem(sys.modules, "max", max_module)
    monkeypatch.setitem(sys.modules, "max.entrypoints", entrypoints_module)
    monkeypatch.setitem(sys.modules, "max.entrypoints.llm", entrypoints_llm_module)
    monkeypatch.setitem(sys.modules, "max.pipelines", pipelines_module)



def _init_model_with_fake_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, llm_cls: type | None = None
) -> object:
    _install_fake_max_modules(monkeypatch, llm_cls)
    return _core.init_model(str(tmp_path))


def test_mojo_init_model_rejects_empty_path() -> None:
    with pytest.raises(Exception, match="non-empty model_path"):
        _core.init_model("")


def test_mojo_init_model_rejects_missing_path() -> None:
    missing_path = Path("/does/not/exist")

    with pytest.raises(Exception, match="does not exist"):
        _core.init_model(str(missing_path))


def test_mojo_step_contract_is_token_sensitive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _init_model_with_fake_runtime(tmp_path, monkeypatch)

    logits_for_token_one: npt.NDArray[np.float32] = np.asarray(
        _core.step(llm, 11, 1.0, 10, 1.0), dtype=np.float32
    )
    logits_for_token_two: npt.NDArray[np.float32] = np.asarray(
        _core.step(llm, 22, 1.0, 10, 1.0), dtype=np.float32
    )

    assert logits_for_token_one.shape == logits_for_token_two.shape
    assert logits_for_token_one.ndim == 1
    assert logits_for_token_one.dtype == np.float32
    assert not np.array_equal(logits_for_token_one, logits_for_token_two)


def test_mojo_embeddings_contract_enforces_shape_dtype_and_determinism(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _init_model_with_fake_runtime(tmp_path, monkeypatch)
    tokens_a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    tokens_b = np.array([[7, 8, 9]], dtype=np.int32)

    emb_a_first = np.asarray(_core.generate_embeddings(llm, tokens_a), dtype=np.float32)
    emb_a_second = np.asarray(_core.generate_embeddings(llm, tokens_a), dtype=np.float32)
    emb_b = np.asarray(_core.generate_embeddings(llm, tokens_b), dtype=np.float32)

    assert emb_a_first.ndim == 2
    assert emb_a_first.shape == (2, 768)
    assert emb_a_first.dtype == np.float32
    assert np.array_equal(emb_a_first, emb_a_second)
    assert not np.array_equal(emb_a_first[0], emb_b[0])


def test_mojo_step_contract_rejects_non_vector_backend_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidStepLLM:
        def __init__(self, pipeline_config: object) -> None:
            self.pipeline_config = pipeline_config

        def step(self, token_id: int, temp: float, top_k: int, top_p: float) -> npt.NDArray[np.float64]:
            del token_id, temp, top_k, top_p
            return np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)

    llm = _init_model_with_fake_runtime(tmp_path, monkeypatch, InvalidStepLLM)

    with pytest.raises(Exception, match="1D"):
        _core.step(llm, 11, 1.0, 10, 1.0)


def test_mojo_embeddings_contract_rejects_non_768d_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidEmbeddingLLM:
        def __init__(self, pipeline_config: object) -> None:
            self.pipeline_config = pipeline_config

        def encode(self, input_array: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
            del input_array
            return np.zeros((1, 128), dtype=np.float32)

    llm = _init_model_with_fake_runtime(tmp_path, monkeypatch, InvalidEmbeddingLLM)

    with pytest.raises(Exception, match="768"):
        _core.generate_embeddings(llm, np.array([[1, 2, 3]], dtype=np.int32))


def test_mojo_init_model_rejects_file_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_path = tmp_path / "not-a-directory.txt"
    file_path.write_text("not a model directory")

    _install_fake_max_modules(monkeypatch)

    with pytest.raises(Exception, match="is not a directory"):
        _core.init_model(str(file_path))
