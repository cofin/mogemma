from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest

import mogemma.model as model_module
from mogemma import EmbeddingConfig, EmbeddingModel


@pytest.fixture
def dummy_model_path(tmp_path: Path) -> str:
    """Fixture for a dummy model path."""
    model_dir = tmp_path / "bert-base-uncased"
    model_dir.mkdir()
    return str(model_dir)


@pytest.fixture
def mock_tokenizer() -> Iterator[MagicMock]:
    """Fixture to mock the AutoTokenizer."""
    with patch("mogemma.model._TokenizerImpl.from_pretrained") as mock:
        tokenizer = MagicMock()

        def _encode_batch(inputs: str | list[str], **_: object) -> list[MagicMock]:
            batch_size = len(inputs) if isinstance(inputs, list) else 1
            result = []
            for _i in range(batch_size):
                encoded = MagicMock()
                encoded.ids = [1, 2, 3]
                result.append(encoded)
            return result

        tokenizer.encode_batch.side_effect = _encode_batch
        mock.return_value = tokenizer
        yield tokenizer


@pytest.fixture
def mock_core(monkeypatch: pytest.MonkeyPatch) -> object:
    """Fixture to mock Mojo core embedding calls."""

    class CoreStub:
        def init_model(self, _: str) -> object:
            return object()

        def generate_embeddings(self, llm: object, tokens: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
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


def test_embedding_model_init(
    dummy_model_path: str, mock_tokenizer: MagicMock, mock_core: object
) -> None:
    """Test embedding model initialization."""
    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)
    assert model is not None


def test_embedding_model_init_rejects_unknown_local_path() -> None:
    config = EmbeddingConfig(model_path="bert-base-uncased-missing")

    with pytest.raises(ValueError, match="existing local directory"):
        EmbeddingModel(config)


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
    texts = ["Hello", "Mojo runs Gemma inference."]
    embeddings = model.embed(texts)

    assert embeddings.shape == (2, 768)


def test_embed_raises_when_backend_returns_wrong_row_count(
    dummy_model_path: str, mock_tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure backend row-count mismatches are not silently accepted."""

    class CoreStub:
        def init_model(self, _: str) -> object:
            return object()

        def generate_embeddings(self, llm: object, tokens: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
            del llm, tokens
            return np.zeros((1, 768), dtype=np.float32)

    monkeypatch.setattr(model_module, "_core", CoreStub())

    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)

    with pytest.raises(ValueError, match="embedding rows"):
        model.embed(["first", "second"])


def test_embed_tokens_uses_mojo_without_tokenizer(dummy_model_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure pre-tokenized embedding path does not require transformers."""

    class CoreStub:
        def init_model(self, _: str) -> object:
            return object()

        def generate_embeddings(self, llm: object, tokens: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
            del llm
            return np.ones((tokens.shape[0], 768), dtype=np.float32)

    monkeypatch.setattr(model_module, "_core", CoreStub())
    monkeypatch.setattr(model_module, "_TokenizerImpl", None)

    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)
    embeddings = model.embed_tokens(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))

    assert embeddings.shape == (2, 768)
    assert embeddings.dtype == np.float32


def test_embed_text_requires_tokenizer_when_transformers_missing(
    dummy_model_path: str, monkeypatch: pytest.MonkeyPatch, mock_core: object
) -> None:
    """Ensure text embedding reports clear requirement when transformers is absent."""
    monkeypatch.setattr(model_module, "_TokenizerImpl", None)

    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)

    with pytest.raises(ModuleNotFoundError, match="embed_tokens"):
        model.embed("hello")


def test_embed_requires_mojo_core(
    dummy_model_path: str, mock_tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure embedding never falls back to non-Mojo output when core is missing."""
    monkeypatch.setattr(model_module, "_core", None)

    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    with pytest.raises(RuntimeError, match="Mojo core is unavailable"):
        EmbeddingModel(config)


def test_embedding_init_raises_on_core_init_failure(
    dummy_model_path: str, mock_tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    class CoreInitFailure:
        def init_model(self, _: str) -> object:
            msg = "checkpoint missing or unreadable"
            raise RuntimeError(msg)

    monkeypatch.setattr(model_module, "_core", CoreInitFailure())
    config = EmbeddingConfig(model_path=Path(dummy_model_path))

    with pytest.raises(RuntimeError, match="embedding model failed to initialize"):
        EmbeddingModel(config)
