import pytest
from pathlib import Path
from mogemma import GenerationConfig, GemmaModel

@pytest.fixture
def dummy_model_path():
    return "bert-base-uncased"

def test_generation_config_validation():
    with pytest.raises(ValueError, match="temperature"):
        GenerationConfig(model_path="dummy", temperature=-1.0)
    with pytest.raises(ValueError, match="top_p"):
        GenerationConfig(model_path="dummy", top_p=1.5)

def test_gemma_model_init(dummy_model_path):
    config = GenerationConfig(model_path=dummy_model_path)
    model = GemmaModel(config)
    assert model is not None

def test_gemma_generate_sampling(dummy_model_path):
    config = GenerationConfig(
        model_path=Path(dummy_model_path), 
        max_new_tokens=5,
        temperature=0.7,
        top_k=10,
        top_p=0.9
    )
    model = GemmaModel(config)
    
    response = model.generate("Hello")
    assert isinstance(response, str)
    assert len(response) > 0
