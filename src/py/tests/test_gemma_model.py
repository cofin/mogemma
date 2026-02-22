from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest

import mogemma.model as model_module
from mogemma import GenerationConfig, SyncGemmaModel


class CoreStub:
    """Stub for the Mojo _core module used in tests."""

    def __init__(self) -> None:
        self.step_calls: list[tuple[int, float, int, float]] = []

    def init_model(self, _: str) -> object:
        return object()

    def step(self, llm: object, token_id: int, temp: float, top_k: int, top_p: float) -> npt.NDArray[np.float32]:
        del llm
        self.step_calls.append((token_id, temp, top_k, top_p))
        # Return logits that always select token 0
        return np.array([5.0, 0.0, 0.0], dtype=np.float32)


@pytest.fixture
def dummy_model_path() -> str:
    return "bert-base-uncased"


@pytest.fixture
def mock_tokenizer() -> Iterator[MagicMock]:
    with patch("mogemma.model.Tokenizer.from_pretrained") as mock:
        tokenizer = MagicMock()
        encoded_mock = MagicMock()
        encoded_mock.ids = [1, 2, 3]
        tokenizer.encode.return_value = encoded_mock
        tokenizer.decode.return_value = "decoded text"
        mock.return_value = tokenizer
        yield tokenizer


@pytest.fixture
def mock_core(monkeypatch: pytest.MonkeyPatch) -> CoreStub:
    stub = CoreStub()
    monkeypatch.setattr(model_module, "_core", stub)
    return stub


def test_generation_config_validation() -> None:
    with pytest.raises(ValueError, match="temperature"):
        GenerationConfig(model_path="dummy", temperature=-1.0)
    with pytest.raises(ValueError, match="top_p"):
        GenerationConfig(model_path="dummy", top_p=1.5)


def test_gemma_model_init(dummy_model_path: str, mock_tokenizer: MagicMock) -> None:
    config = GenerationConfig(model_path=Path(dummy_model_path))
    model = SyncGemmaModel(config)
    assert model is not None


def test_gemma_generate_sampling(dummy_model_path: str, mock_tokenizer: MagicMock, mock_core: CoreStub) -> None:
    config = GenerationConfig(model_path=Path(dummy_model_path), max_new_tokens=5, temperature=0.7, top_k=10, top_p=0.9)
    model = SyncGemmaModel(config)

    response = model.generate("Hello")
    assert isinstance(response, str)
    assert len(response) > 0


def test_gemma_generate_long_prompt(dummy_model_path: str, mock_tokenizer: MagicMock, mock_core: CoreStub) -> None:
    config = GenerationConfig(model_path=Path(dummy_model_path), max_sequence_length=1024)
    model = SyncGemmaModel(config)
    response = model.generate("word " * 500)
    assert isinstance(response, str)


def test_gemma_generate_empty_prompt(dummy_model_path: str, mock_tokenizer: MagicMock, mock_core: CoreStub) -> None:
    config = GenerationConfig(model_path=Path(dummy_model_path))
    model = SyncGemmaModel(config)
    response = model.generate("")
    assert isinstance(response, str)


def test_gemma_consecutive_generations(dummy_model_path: str, mock_tokenizer: MagicMock, mock_core: CoreStub) -> None:
    config = GenerationConfig(model_path=Path(dummy_model_path), max_new_tokens=5)
    model = SyncGemmaModel(config)

    res1 = model.generate("First prompt")
    res2 = model.generate("Second prompt")

    assert isinstance(res1, str)
    assert isinstance(res2, str)
    assert len(res1) > 0
    assert len(res2) > 0


def test_gemma_generate_stream_uses_backend_logits(
    dummy_model_path: str, mock_tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure each generated token is selected from backend logits."""

    class PreciseStub:
        def __init__(self) -> None:
            self.step_calls: list[tuple[int, float, int, float]] = []

        def init_model(self, _: str) -> object:
            return object()

        def step(self, llm: object, token_id: int, temp: float, top_k: int, top_p: float) -> npt.NDArray[np.float32]:
            del llm
            self.step_calls.append((token_id, temp, top_k, top_p))
            if len(self.step_calls) == 1:
                return np.array([0.0, 0.0, 5.0], dtype=np.float32)
            return np.array([4.0, 0.0, 0.0], dtype=np.float32)

    core_stub = PreciseStub()
    monkeypatch.setattr(model_module, "_core", core_stub)

    mock_tokenizer.decode.side_effect = lambda token_ids: f"<{token_ids[0]}>"

    config = GenerationConfig(model_path=Path(dummy_model_path), max_new_tokens=2, temperature=0.0, top_k=50, top_p=1.0)
    model = SyncGemmaModel(config)

    output = model.generate("Hello")
    assert output == "<2><0>"
    assert core_stub.step_calls == [(3, 0.0, 50, 1.0), (2, 0.0, 50, 1.0)]


def test_gemma_generate_raises_without_core(
    dummy_model_path: str, mock_tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_module, "_core", None)
    config = GenerationConfig(model_path=Path(dummy_model_path))
    model = SyncGemmaModel(config)
    with pytest.raises(RuntimeError, match="Mojo core is unavailable"):
        model.generate("Hello")
