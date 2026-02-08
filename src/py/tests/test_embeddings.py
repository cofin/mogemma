import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from mogemma import EmbeddingConfig, EmbeddingModel

@pytest.fixture
def dummy_model_path():
    return "bert-base-uncased"

@pytest.fixture
def mock_tokenizer():
    with patch("mogemma.model.AutoTokenizer.from_pretrained") as mock:
        tokenizer = MagicMock()
        # Mock encoding (calling the tokenizer object)
        tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]], dtype=np.int32)
        }
        mock.return_value = tokenizer
        yield tokenizer

def test_embedding_config(dummy_model_path):
    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    assert str(config.model_path) == dummy_model_path
    assert config.device == "cpu"

def test_embedding_model_init(dummy_model_path, mock_tokenizer):
    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)
    assert model is not None

def test_embed_single_string(dummy_model_path, mock_tokenizer):
    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)
    embeddings = model.embed("Hello world")
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (1, 768)
    assert embeddings.dtype == np.float32

def test_embed_list_of_strings(dummy_model_path, mock_tokenizer):
    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)
    texts = ["Hello", "Mojo is fast"]
    embeddings = model.embed(texts)
    
    assert embeddings.shape == (2, 768) or embeddings.shape == (1, 768)
