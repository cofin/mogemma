from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mogemma import GenerationConfig
from mogemma.async_model import AsyncGemmaModel


@pytest.fixture
def mock_tokenizer():
    with patch("mogemma.model.AutoTokenizer.from_pretrained") as mock:
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "token "
        tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]]),
            "attention_mask": np.array([[1, 1, 1]])
        }
        mock.return_value = tokenizer
        yield tokenizer

@pytest.mark.asyncio
async def test_async_generate(tmp_path, mock_tokenizer):
    """Verify async generation returns a string."""
    model_dir = tmp_path / "dummy-model"
    model_dir.mkdir()

    config = GenerationConfig(model_path=model_dir, max_new_tokens=5)
    model = AsyncGemmaModel(config)

    response = await model.generate("Hello")
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.asyncio
async def test_async_generate_stream(tmp_path, mock_tokenizer):
    """Verify async streaming yields tokens."""
    model_dir = tmp_path / "dummy-model"
    model_dir.mkdir()

    config = GenerationConfig(model_path=model_dir, max_new_tokens=5)
    model = AsyncGemmaModel(config)

    tokens = []
    async for token in model.generate_stream("Hello"):
        tokens.append(token)

    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)
