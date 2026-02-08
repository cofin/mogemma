import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from mogemma import GenerationConfig, GemmaModel

@pytest.fixture
def dummy_model_path():
    return "bert-base-uncased"

@pytest.fixture
def mock_tokenizer():
    with patch("mogemma.model.AutoTokenizer.from_pretrained") as mock:
        tokenizer = MagicMock()
        # Mock encoding
        tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]], dtype=np.int32)
        }
        # Mock decoding
        tokenizer.decode.return_value = "decoded text"
        mock.return_value = tokenizer
        yield tokenizer

def test_generation_config_validation():
    with pytest.raises(ValueError, match="temperature"):
        GenerationConfig(model_path="dummy", temperature=-1.0)
    with pytest.raises(ValueError, match="top_p"):
        GenerationConfig(model_path="dummy", top_p=1.5)

def test_gemma_model_init(dummy_model_path, mock_tokenizer):
    config = GenerationConfig(model_path=Path(dummy_model_path))
    model = GemmaModel(config)
    assert model is not None

def test_gemma_generate_sampling(dummy_model_path, mock_tokenizer):
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
