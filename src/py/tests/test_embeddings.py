from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import mogemma.model as model_module
from mogemma import EmbeddingConfig, EmbeddingModel


@pytest.fixture
def dummy_model_path() -> str:
    """Fixture for a dummy model path."""
    return "bert-base-uncased"


@pytest.fixture
def mock_tokenizer() -> Iterator[MagicMock]:
    """Fixture to mock the AutoTokenizer."""
    with patch("mogemma.model.AutoTokenizer.from_pretrained") as mock:
        tokenizer = MagicMock()

        def _encode(inputs: str | list[str], **_: object) -> dict[str, np.ndarray]:
            batch_size = len(inputs) if isinstance(inputs, list) else 1
            input_ids = np.tile(np.array([[1, 2, 3]], dtype=np.int32), (batch_size, 1))
            return {"input_ids": input_ids}

        tokenizer.side_effect = _encode
        mock.return_value = tokenizer
        yield tokenizer


@pytest.fixture
def mock_core(monkeypatch: pytest.MonkeyPatch) -> object:
    """Fixture to mock Mojo core embedding calls."""

    class CoreStub:
        def init_model(self, _: str) -> object:
            return object()

        def generate_embeddings(self, llm: object, tokens: np.ndarray) -> np.ndarray:
            del llm
            return np.ones((tokens.shape[0], 768), dtype=np.float32)

    core_stub = CoreStub()
    monkeypatch.setattr(model_module, "_core", core_stub)
    return core_stub


def test_embedding_config(dummy_model_path: str) -> None:
    """Test embedding configuration initialization."""
    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    assert str(config.model_path) == dummy_model_path
    assert config.device == "cpu"


def test_embedding_model_init(dummy_model_path: str, mock_tokenizer: MagicMock) -> None:
    """Test embedding model initialization."""
    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)
    assert model is not None


def test_embed_single_string(dummy_model_path: str, mock_tokenizer: MagicMock, mock_core: object) -> None:
    """Test embedding a single string."""
    del mock_core
    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)
    embeddings = model.embed("Hello world")

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (1, 768)
    assert embeddings.dtype == np.float32


def test_embed_list_of_strings(dummy_model_path: str, mock_tokenizer: MagicMock, mock_core: object) -> None:
    """Test embedding a list of strings."""
    del mock_core
    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)
    texts = ["Hello", "Mojo is fast"]
    embeddings = model.embed(texts)

    assert embeddings.shape == (2, 768)


def test_embed_raises_when_backend_returns_wrong_row_count(
    dummy_model_path: str,
    mock_tokenizer: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure backend row-count mismatches are not silently accepted."""

    class CoreStub:
        def init_model(self, _: str) -> object:
            return object()

        def generate_embeddings(self, llm: object, tokens: np.ndarray) -> np.ndarray:
            del llm, tokens
            return np.zeros((1, 768), dtype=np.float32)

    monkeypatch.setattr(model_module, "_core", CoreStub())

    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)

    with pytest.raises(ValueError, match="embedding rows"):
        model.embed(["first", "second"])


def test_embed_tokens_uses_mojo_without_tokenizer(
    dummy_model_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure pre-tokenized embedding path does not require transformers."""

    class CoreStub:
        def init_model(self, _: str) -> object:
            return object()

        def generate_embeddings(self, llm: object, tokens: np.ndarray) -> np.ndarray:
            del llm
            return np.ones((tokens.shape[0], 768), dtype=np.float32)

    monkeypatch.setattr(model_module, "_core", CoreStub())
    monkeypatch.setattr(model_module, "AutoTokenizer", None)

    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)
    embeddings = model.embed_tokens(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))

    assert embeddings.shape == (2, 768)
    assert embeddings.dtype == np.float32


def test_embed_text_requires_tokenizer_when_transformers_missing(
    dummy_model_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure text embedding reports clear requirement when transformers is absent."""
    monkeypatch.setattr(model_module, "AutoTokenizer", None)

    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)

    with pytest.raises(ModuleNotFoundError, match="embed_tokens"):
        model.embed("hello")


def test_embed_requires_mojo_core(
    dummy_model_path: str,
    mock_tokenizer: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure embedding never falls back to non-Mojo output when core is missing."""
    monkeypatch.setattr(model_module, "_core", None)

    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)

    with pytest.raises(RuntimeError, match="Mojo core is unavailable"):
        model.embed("hello")
