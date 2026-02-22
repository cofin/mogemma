from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mogemma import GenerationConfig
from mogemma.vision_model import VisionGemmaModel


@pytest.fixture
def dummy_model_path():
    return "bert-base-uncased"


@pytest.fixture
def mock_tokenizer():
    with patch("mogemma.model.Tokenizer.from_pretrained") as mock:
        tokenizer = MagicMock()
        encoded_mock = MagicMock()
        encoded_mock.ids = [1, 2, 3]
        tokenizer.encode.return_value = encoded_mock
        tokenizer.decode.return_value = "A cat"
        mock.return_value = tokenizer
        yield tokenizer


def test_vision_model_init(dummy_model_path, mock_tokenizer):
    config = GenerationConfig(model_path=Path(dummy_model_path))
    model = VisionGemmaModel(config)
    assert model is not None


def test_generate_from_image(dummy_model_path, mock_tokenizer):
    config = GenerationConfig(model_path=Path(dummy_model_path))
    model = VisionGemmaModel(config)

    # Simulate a dummy image
    image = np.zeros((224, 224, 3), dtype=np.uint8)


def test_generate_interleaved(dummy_model_path, mock_tokenizer):
    config = GenerationConfig(model_path=Path(dummy_model_path))
    model = VisionGemmaModel(config)

    image = np.zeros((224, 224, 3), dtype=np.uint8)

    # List of (type, content) tuples or just mixed content
    # For MVP, let's say we pass a list: ["Look at this:", image, "What is it?"]
    prompt = ["Look at this:", image, "What is it?"]

    response = model.generate_multimodal(prompt)
    assert isinstance(response, str)
