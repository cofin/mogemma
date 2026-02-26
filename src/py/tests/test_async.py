import json
import struct
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest

import mogemma.model as model_module
from mogemma import GenerationConfig
from mogemma.model import AsyncGemmaModel


def _create_dummy_safetensors(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "model.safetensors").open("wb") as f:
        h = json.dumps({}).encode("utf-8")
        f.write(struct.pack("<Q", len(h)) + h)
    (model_dir / "tokenizer.model").touch()


class CoreStub:
    def init_model(self, _: str) -> object:
        return object()

    def step(self, llm: object, token_id: int, temp: float, top_k: int, top_p: float) -> npt.NDArray[np.float32]:
        del llm, token_id, temp, top_k, top_p
        return np.array([0.0, 5.0, 0.0], dtype=np.float32)


@pytest.fixture
def mock_tokenizer() -> Iterator[MagicMock]:
    with patch("mogemma.model._Tokenizer") as mock:
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "token "
        tokenizer.token_to_id.return_value = None

        encoded_mock = MagicMock()
        encoded_mock.ids = [1, 2, 3]
        tokenizer.encode.return_value = encoded_mock

        mock.return_value = tokenizer
        yield tokenizer


@pytest.fixture
def mock_core(monkeypatch: pytest.MonkeyPatch) -> CoreStub:
    stub = CoreStub()
    monkeypatch.setattr(model_module, "_core", stub)
    return stub


@pytest.mark.asyncio
async def test_async_generate(tmp_path: Path, mock_tokenizer: MagicMock, mock_core: CoreStub) -> None:
    model_dir = tmp_path / "dummy-model"
    _create_dummy_safetensors(model_dir)

    config = GenerationConfig(model_path=model_dir, max_tokens=5, temperature=0.0)
    model = AsyncGemmaModel(config)

    response = await model.generate("Hello")
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.asyncio
async def test_async_generate_stream(tmp_path: Path, mock_tokenizer: MagicMock, mock_core: CoreStub) -> None:
    model_dir = tmp_path / "dummy-model"
    _create_dummy_safetensors(model_dir)

    config = GenerationConfig(model_path=model_dir, max_tokens=5, temperature=0.0)
    model = AsyncGemmaModel(config)

    tokens = [token async for token in model.generate_stream("Hello")]

    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)


@pytest.mark.asyncio
async def test_async_generate_stream_stops_on_eos(
    tmp_path: Path, mock_tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    class EOSStub(CoreStub):
        def step(self, llm: object, token_id: int, temp: float, top_k: int, top_p: float) -> npt.NDArray[np.float32]:
            del llm, token_id, temp, top_k, top_p
            return np.array([5.0, 0.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(model_module, "_core", EOSStub())
    mock_tokenizer.token_to_id.return_value = 0

    model_dir = tmp_path / "dummy-model"
    _create_dummy_safetensors(model_dir)
    config = GenerationConfig(model_path=model_dir, max_tokens=5, temperature=0.0)
    model = AsyncGemmaModel(config)

    tokens = [token async for token in model.generate_stream("Hello")]

    assert tokens == []
