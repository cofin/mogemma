import json
import struct
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest

import mogemma.model as model_module
from mogemma import EmbeddingConfig, EmbeddingModel, GenerationConfig, SyncGemmaModel


def _create_dummy_safetensors(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "model.safetensors").open("wb") as f:
        h = json.dumps({}).encode("utf-8")
        f.write(struct.pack("<Q", len(h)) + h)
    (model_dir / "tokenizer.model").touch()


@pytest.fixture
def mock_generation_tokenizer() -> Iterator[MagicMock]:
    """Patch tokenizer for generation-style calls."""
    with patch("mogemma.model._Tokenizer") as mock:
        tokenizer = MagicMock()

        encoded = MagicMock()
        encoded.ids = [11]
        tokenizer.encode.return_value = encoded
        tokenizer.decode.return_value = "<tok>"
        tokenizer.enable_truncation.return_value = None
        tokenizer.enable_padding.return_value = None
        tokenizer.token_to_id.return_value = 0

        mock.return_value = tokenizer
        yield tokenizer


@pytest.fixture
def mock_embedding_tokenizer() -> Iterator[MagicMock]:
    """Patch tokenizer for text embedding calls."""
    with patch("mogemma.model._Tokenizer") as mock:
        tokenizer = MagicMock()

        def _encode_batch(texts: list[str], **_: object) -> list[MagicMock]:
            encoded = []
            for _text in texts:
                tokenized = MagicMock()
                tokenized.ids = [1, 2, 3]
                encoded.append(tokenized)
            return encoded

        tokenizer.encode_batch.side_effect = _encode_batch
        tokenizer.enable_truncation.return_value = None
        tokenizer.enable_padding.return_value = None
        tokenizer.token_to_id.return_value = 0

        mock.return_value = tokenizer
        yield tokenizer


@pytest.fixture
def mock_core_init_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force model initialization failure for runtime contract checks."""

    class CoreInitFailure:
        def init_model(self, _: str) -> object:
            msg = "checkpoint missing or unreadable"
            raise RuntimeError(msg)

    monkeypatch.setattr(model_module, "_core", CoreInitFailure())


@pytest.fixture
def mock_generation_empty_logits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force a deterministic empty-step-failure path."""

    class CoreEmptyLogits:
        def init_model(self, _: str) -> object:
            return object()

        def step(self, llm: object, token_id: int, temp: float, top_k: int, top_p: float) -> npt.NDArray[np.float32]:
            del llm, token_id, temp, top_k, top_p
            return np.array([], dtype=np.float32)

    monkeypatch.setattr(model_module, "_core", CoreEmptyLogits())


@pytest.fixture
def mock_embedding_non_matrix_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force non-matrix embedding output from core."""

    class CoreNonMatrix:
        def init_model(self, _: str) -> object:
            return object()

        def generate_embeddings(self, llm: object, tokens: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
            del llm, tokens
            return np.zeros((768,), dtype=np.float32)

    monkeypatch.setattr(model_module, "_core", CoreNonMatrix())


@pytest.fixture
def mock_embedding_float64_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force backend float64 embeddings to validate casting contract."""

    class CoreFloat64:
        def init_model(self, _: str) -> object:
            return object()

        def generate_embeddings(self, llm: object, tokens: list[list[int]]) -> npt.NDArray[np.float64]:
            del llm
            return np.ones((len(tokens), 768), dtype=np.float64)

    monkeypatch.setattr(model_module, "_core", CoreFloat64())


def test_generation_init_propagates_contract_error(
    tmp_path: Path, mock_generation_tokenizer: MagicMock, mock_core_init_failure: None
) -> None:
    _create_dummy_safetensors(tmp_path)
    config = GenerationConfig(model_path=tmp_path)

    with pytest.raises(RuntimeError, match="generation model failed to initialize"):
        SyncGemmaModel(config)


def test_generation_empty_logits_is_deterministic_failure(
    tmp_path: Path, mock_generation_tokenizer: MagicMock, mock_generation_empty_logits: None
) -> None:
    _create_dummy_safetensors(tmp_path)
    config = GenerationConfig(model_path=tmp_path, max_new_tokens=1)
    model = SyncGemmaModel(config)

    with pytest.raises(ValueError, match="backend returned empty logits"):
        model.generate("hello")


def test_embedding_init_propagates_contract_error(
    tmp_path: Path, mock_embedding_tokenizer: MagicMock, mock_core_init_failure: None
) -> None:
    _create_dummy_safetensors(tmp_path)
    config = EmbeddingConfig(model_path=tmp_path)

    with pytest.raises(RuntimeError, match="embedding model failed to initialize"):
        EmbeddingModel(config)


def test_embed_tokens_rejects_non_matrix_backend_output(tmp_path: Path, mock_embedding_non_matrix_output: None) -> None:
    _create_dummy_safetensors(tmp_path)
    config = EmbeddingConfig(model_path=tmp_path)
    model = EmbeddingModel(config)
    tokens = np.array([[1, 2, 3]], dtype=np.int32)

    with pytest.raises(ValueError, match="expected 2D embeddings"):
        model.embed_tokens(tokens)


def test_embed_tokens_casts_to_float32(tmp_path: Path, mock_embedding_float64_output: None) -> None:
    _create_dummy_safetensors(tmp_path)
    config = EmbeddingConfig(model_path=tmp_path)
    model = EmbeddingModel(config)
    tokens = np.array([[1, 2, 3]], dtype=np.int32)

    embeddings = model.embed_tokens(tokens)

    assert embeddings.dtype == np.float32
    assert embeddings.shape == (1, 768)
