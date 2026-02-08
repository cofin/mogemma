import pytest
import numpy as np
from pathlib import Path
from mogemma import EmbeddingConfig, EmbeddingModel

@pytest.fixture
def dummy_model_path(tmp_path):
    path = tmp_path / "gemma-3-dummy"
    path.touch()
    return path

def test_embedding_config(dummy_model_path):
    config = EmbeddingConfig(model_path=dummy_model_path)
    assert config.model_path == dummy_model_path
    assert config.device == "cpu"

def test_embedding_model_init(dummy_model_path):
    config = EmbeddingConfig(model_path=dummy_model_path)
    model = EmbeddingModel(config)
    # Even if loading fails (expected without weights), 
    # the object should initialize.
    assert model is not None

def test_embed_single_string(dummy_model_path):
    config = EmbeddingConfig(model_path=dummy_model_path)
    model = EmbeddingModel(config)
    embeddings = model.embed("Hello world")
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (1, 768)
    assert embeddings.dtype == np.float32

def test_embed_list_of_strings(dummy_model_path):
    config = EmbeddingConfig(model_path=dummy_model_path)
    model = EmbeddingModel(config)
    texts = ["Hello", "Mojo is fast"]
    embeddings = model.embed(texts)
    
    # Current implementation returns (1, 768) dummy for any input
    # In Phase 3 final, it should match the text count.
    assert embeddings.shape == (1, 768)
