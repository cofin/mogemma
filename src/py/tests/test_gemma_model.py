from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mogemma import GemmaModel, GenerationConfig


@pytest.fixture
def dummy_model_path() -> str:
    """Fixture for a dummy model path."""
    return "bert-base-uncased"


@pytest.fixture
def mock_tokenizer() -> Iterator[MagicMock]:
    """Fixture to mock the AutoTokenizer."""
    with patch("mogemma.model.AutoTokenizer.from_pretrained") as mock:
        tokenizer = MagicMock()
        # Mock encoding
        tokenizer.return_value = {"input_ids": np.array([[1, 2, 3]], dtype=np.int32)}
        # Mock decoding
        tokenizer.decode.return_value = "decoded text"
        mock.return_value = tokenizer
        yield tokenizer


def test_generation_config_validation() -> None:
    """Test generation configuration validation logic."""
    with pytest.raises(ValueError, match="temperature"):
        GenerationConfig(model_path="dummy", temperature=-1.0)
    with pytest.raises(ValueError, match="top_p"):
        GenerationConfig(model_path="dummy", top_p=1.5)


def test_gemma_model_init(dummy_model_path: str, mock_tokenizer: MagicMock) -> None:
    """Test text generation model initialization."""
    config = GenerationConfig(model_path=Path(dummy_model_path))
    model = GemmaModel(config)
    assert model is not None


def test_gemma_generate_sampling(dummy_model_path: str, mock_tokenizer: MagicMock) -> None:
    """Test text generation with sampling parameters."""
    config = GenerationConfig(model_path=Path(dummy_model_path), max_new_tokens=5, temperature=0.7, top_k=10, top_p=0.9)
    model = GemmaModel(config)

    response = model.generate("Hello")
    assert isinstance(response, str)
    assert len(response) > 0


def test_gemma_generate_long_prompt(dummy_model_path: str, mock_tokenizer: MagicMock) -> None:
    """Test text generation with a long prompt."""
    config = GenerationConfig(model_path=Path(dummy_model_path), max_sequence_length=1024)
    model = GemmaModel(config)
    long_prompt = "word " * 500
    response = model.generate(long_prompt)
    assert isinstance(response, str)


def test_gemma_generate_empty_prompt(dummy_model_path: str, mock_tokenizer: MagicMock) -> None:
    """Test text generation with an empty prompt."""
    config = GenerationConfig(model_path=Path(dummy_model_path))
    model = GemmaModel(config)
    response = model.generate("")
    assert isinstance(response, str)


def test_gemma_consecutive_generations(dummy_model_path: str, mock_tokenizer: MagicMock) -> None:
    """Test consecutive text generations."""
    config = GenerationConfig(model_path=Path(dummy_model_path), max_new_tokens=5)
    model = GemmaModel(config)

    res1 = model.generate("First prompt")
    res2 = model.generate("Second prompt")

    assert res1 == res2  # In our mock/dummy mode, they should be the same
