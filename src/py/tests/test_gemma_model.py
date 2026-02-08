import pytest
from pathlib import Path
from mogemma import GenerationConfig, GemmaModel

@pytest.fixture
def dummy_model_path():
    return "bert-base-uncased"

def test_generation_config(dummy_model_path):
    config = GenerationConfig(model_path=Path(dummy_model_path))
    assert str(config.model_path) == dummy_model_path
    assert config.temperature == 1.0

def test_gemma_model_init(dummy_model_path):
    config = GenerationConfig(model_path=dummy_model_path)
    model = GemmaModel(config)
    assert model is not None

def test_gemma_generate(dummy_model_path):
    config = GenerationConfig(model_path=dummy_model_path)
    model = GemmaModel(config)
    response = model.generate("What is Mojo?")
    assert isinstance(response, str)
    assert len(response) > 0
