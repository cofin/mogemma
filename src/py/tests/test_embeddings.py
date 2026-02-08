from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

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
        # Mock encoding (calling the tokenizer object)
        tokenizer.return_value = {"input_ids": np.array([[1, 2, 3]], dtype=np.int32)}
        mock.return_value = tokenizer
        yield tokenizer


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


def test_embed_single_string(dummy_model_path: str, mock_tokenizer: MagicMock) -> None:
    """Test embedding a single string."""
    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)
    embeddings = model.embed("Hello world")

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (1, 768)
    assert embeddings.dtype == np.float32


def test_embed_list_of_strings(dummy_model_path: str, mock_tokenizer: MagicMock) -> None:
    """Test embedding a list of strings."""
    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)
    texts = ["Hello", "Mojo is fast"]
    embeddings = model.embed(texts)

    assert embeddings.shape in ((2, 768), (1, 768))
