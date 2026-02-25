import json
import struct
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest

import mogemma.model as model_module
from mogemma import GenerationConfig, SyncGemmaModel
from mogemma.hub import HubManager


def _create_dummy_safetensors(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "model.safetensors").open("wb") as f:
        h = json.dumps({}).encode("utf-8")
        f.write(struct.pack("<Q", len(h)) + h)
    # Also need a dummy tokenizer.model for _ensure_tokenizer
    (model_dir / "tokenizer.model").touch()


class CoreStub:
    """Stub for the Mojo _core module used in tests."""

    def __init__(self) -> None:
        self.step_calls: list[tuple[int, float, int, float]] = []

    def init_model(self, _: str) -> object:
        return object()

    def step(self, llm: object, token_id: int, temp: float, top_k: int, top_p: float) -> npt.NDArray[np.float32]:
        del llm
        self.step_calls.append((token_id, temp, top_k, top_p))
        # Return logits that always select token 0
        return np.array([5.0, 0.0, 0.0], dtype=np.float32)


@pytest.fixture
def dummy_model_path(tmp_path: Path) -> str:
    model_dir = tmp_path / "bert-base-uncased"
    _create_dummy_safetensors(model_dir)
    return str(model_dir)


@pytest.fixture
def mock_tokenizer() -> Iterator[MagicMock]:
    with patch("mogemma.model._Tokenizer") as mock:
        tokenizer = MagicMock()
        encoded_mock = MagicMock()
        encoded_mock.ids = [1, 2, 3]
        tokenizer.encode.return_value = encoded_mock
        tokenizer.decode.return_value = "decoded text"
        tokenizer.token_to_id.return_value = 999
        mock.return_value = tokenizer
        yield tokenizer


@pytest.fixture
def mock_core(monkeypatch: pytest.MonkeyPatch) -> CoreStub:
    stub = CoreStub()
    monkeypatch.setattr(model_module, "_core", stub)
    return stub


def test_sync_model_default_config(dummy_model_path: str, mock_tokenizer: MagicMock, mock_core: CoreStub) -> None:
    """SyncGemmaModel() with no args should use default GenerationConfig."""
    model = SyncGemmaModel(GenerationConfig(model_path=Path(dummy_model_path)))
    assert model.config.model_path == Path(dummy_model_path)


def test_sync_model_string_config(dummy_model_path: str, mock_tokenizer: MagicMock, mock_core: CoreStub) -> None:
    """SyncGemmaModel(str) should create a config with that model path."""
    model = SyncGemmaModel(dummy_model_path)
    assert str(model.config.model_path) == dummy_model_path


def test_generation_config_validation() -> None:
    with pytest.raises(ValueError, match="temperature"):
        GenerationConfig(model_path="dummy", temperature=-1.0)
    with pytest.raises(ValueError, match="top_p"):
        GenerationConfig(model_path="dummy", top_p=1.5)


def test_gemma_model_init_uses_hub_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_tokenizer: MagicMock, mock_core: CoreStub
) -> None:
    """Model init should resolve through HubManager for HF-style IDs."""
    downloaded = tmp_path / "google--gemma-3-4b-it"
    downloaded.mkdir()
    _create_dummy_safetensors(downloaded)

    called: list[tuple[str, bool, bool]] = []

    def fake_resolve_model(
        self: object, model_id: str, *, download_if_missing: bool, strict: bool, **_: object
    ) -> Path:
        called.append((model_id, download_if_missing, strict))
        return downloaded

    monkeypatch.setattr(HubManager, "resolve_model", fake_resolve_model)

    config = GenerationConfig(model_path="google/gemma-3-4b-it")
    model = SyncGemmaModel(config)

    assert model is not None
    assert called == [("google/gemma-3-4b-it", True, True)]


def test_gemma_model_init(dummy_model_path: str, mock_tokenizer: MagicMock, mock_core: CoreStub) -> None:
    config = GenerationConfig(model_path=Path(dummy_model_path))
    model = SyncGemmaModel(config)
    assert model is not None


def test_gemma_model_init_rejects_unknown_model_path(mock_tokenizer: MagicMock) -> None:
    config = GenerationConfig(model_path="bert-base-uncased-missing")

    with pytest.raises(HubManager.ModelNotFoundError, match="not found in the public gemma-data bucket"):
        SyncGemmaModel(config)


def test_gemma_generate_sampling(dummy_model_path: str, mock_tokenizer: MagicMock, mock_core: CoreStub) -> None:
    config = GenerationConfig(model_path=Path(dummy_model_path), max_tokens=5, temperature=0.7, top_k=10, top_p=0.9)
    model = SyncGemmaModel(config)

    response = model.generate("Hello")
    assert isinstance(response, str)
    assert len(response) > 0


def test_gemma_generate_long_prompt(dummy_model_path: str, mock_tokenizer: MagicMock, mock_core: CoreStub) -> None:
    config = GenerationConfig(model_path=Path(dummy_model_path), max_sequence_length=1024)
    model = SyncGemmaModel(config)
    response = model.generate("word " * 500)
    assert isinstance(response, str)


def test_gemma_generate_empty_prompt(dummy_model_path: str, mock_tokenizer: MagicMock, mock_core: CoreStub) -> None:
    config = GenerationConfig(model_path=Path(dummy_model_path))
    model = SyncGemmaModel(config)
    response = model.generate("")
    assert isinstance(response, str)


def test_gemma_consecutive_generations(dummy_model_path: str, mock_tokenizer: MagicMock, mock_core: CoreStub) -> None:
    config = GenerationConfig(model_path=Path(dummy_model_path), max_tokens=5)
    model = SyncGemmaModel(config)

    res1 = model.generate("First prompt")
    res2 = model.generate("Second prompt")

    assert isinstance(res1, str)
    assert isinstance(res2, str)
    assert len(res1) > 0
    assert len(res2) > 0


def test_gemma_generate_stream_uses_backend_logits(
    dummy_model_path: str, mock_tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure each generated token is selected from backend logits."""

    class PreciseStub:
        def __init__(self) -> None:
            self.step_calls: list[tuple[int, float, int, float]] = []

        def init_model(self, _: str) -> object:
            return object()

        def step(self, llm: object, token_id: int, temp: float, top_k: int, top_p: float) -> npt.NDArray[np.float32]:
            del llm
            self.step_calls.append((token_id, temp, top_k, top_p))
            # prefill includes explicit BOS insertion; generation starts on call 4.
            generation_call_count = 4
            if len(self.step_calls) == generation_call_count:
                return np.array([0.0, 0.0, 5.0], dtype=np.float32)
            return np.array([4.0, 0.0, 0.0], dtype=np.float32)

    core_stub = PreciseStub()
    monkeypatch.setattr(model_module, "_core", core_stub)

    mock_tokenizer.decode.side_effect = lambda token_ids: f"<{token_ids[0]}>"

    config = GenerationConfig(model_path=Path(dummy_model_path), max_tokens=2, temperature=0.0, top_k=50, top_p=1.0)
    model = SyncGemmaModel(config)

    output = model.generate("Hello")
    assert output == "<2><0>"
    assert core_stub.step_calls == [
        (2, 0.0, 50, 1.0),
        (1, 0.0, 50, 1.0),
        (2, 0.0, 50, 1.0),
        (3, 0.0, 50, 1.0),
        (2, 0.0, 50, 1.0),
    ]


def test_gemma_generate_stream_stops_on_eos(
    dummy_model_path: str, mock_tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    class EOSStub:
        def __init__(self) -> None:
            self.step_calls: list[int] = []

        def init_model(self, _: str) -> object:
            return object()

        def step(self, llm: object, token_id: int, temp: float, top_k: int, top_p: float) -> npt.NDArray[np.float32]:
            del llm, token_id, temp, top_k, top_p
            self.step_calls.append(1)
            # len=4 is the first generation call after BOS + prompt prefill.
            generation_call_count = 4
            if len(self.step_calls) == generation_call_count:
                return np.array([5.0, 0.0, 0.0], dtype=np.float32)  # index 0 is eos
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    core_stub = EOSStub()
    monkeypatch.setattr(model_module, "_core", core_stub)
    mock_tokenizer.token_to_id.return_value = 0

    config = GenerationConfig(model_path=Path(dummy_model_path), max_tokens=5, temperature=0.0)
    model = SyncGemmaModel(config)
    response = model.generate("Hello")

    assert response == ""
    assert core_stub.step_calls == [1, 1, 1, 1]


def test_gemma_generate_stream_raises_for_non_string_decode(
    dummy_model_path: str, mock_tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure decoder contract requires string output for decode path."""

    class DecodeTypeStub:
        def __init__(self) -> None:
            self.step_calls = 0

        def init_model(self, _: str) -> object:
            return object()

        def step(self, llm: object, token_id: int, temp: float, top_k: int, top_p: float) -> npt.NDArray[np.float32]:
            del llm, token_id, temp, top_k, top_p
            self.step_calls += 1
            return np.array([0.0, 0.0, 5.0], dtype=np.float32)

    monkeypatch.setattr(model_module, "_core", DecodeTypeStub())
    mock_tokenizer.decode.return_value = b"\xff"

    config = GenerationConfig(model_path=Path(dummy_model_path), max_tokens=1)
    model = SyncGemmaModel(config)

    with pytest.raises(TypeError, match=r"tokenizer\.decode returned non-string output"):
        model.generate("Hello")


def test_gemma_generate_raises_without_core(
    dummy_model_path: str, mock_tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_module, "_core", None)
    config = GenerationConfig(model_path=Path(dummy_model_path))

    with pytest.raises(RuntimeError, match="Mojo core is unavailable"):
        SyncGemmaModel(config)


def test_gemma_init_raises_on_core_init_failure(
    dummy_model_path: str, mock_tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    class CoreInitFailure:
        def init_model(self, _: str) -> object:
            msg = "checkpoint missing or unreadable"
            raise RuntimeError(msg)

    monkeypatch.setattr(model_module, "_core", CoreInitFailure())
    config = GenerationConfig(model_path=Path(dummy_model_path))

    with pytest.raises(RuntimeError, match="generation model failed to initialize"):
        SyncGemmaModel(config)


def test_gemma_generate_applies_instruction_template_for_it_models(
    tmp_path: Path, mock_tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Instruction-tuned models should format raw prompts as user->model turns."""

    class CoreStub:
        def init_model(self, _: str) -> object:
            return {}

        def step(self, llm: object, token_id: int, temp: float, top_k: int, top_p: float) -> npt.NDArray[np.float32]:
            del llm, token_id, temp, top_k, top_p
            return np.array([5.0, 0.0, 0.0], dtype=np.float32)

    model_dir = tmp_path / "gemma3-270m-it"
    _create_dummy_safetensors(model_dir)
    monkeypatch.setattr(model_module, "_core", CoreStub())
    mock_tokenizer.decode.return_value = ""  # stop immediately

    model = SyncGemmaModel(GenerationConfig(model_path=model_dir, max_tokens=1))
    model.generate("What is the capital of France?")

    encoded_prompt = mock_tokenizer.encode.call_args.args[0]
    assert encoded_prompt.startswith("<start_of_turn>user\n")
    assert encoded_prompt.endswith("<start_of_turn>model\n")


def test_gemma_generate_keeps_existing_instruction_template(
    tmp_path: Path, mock_tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Already templated prompts should not be wrapped a second time."""

    class CoreStub:
        def init_model(self, _: str) -> object:
            return {}

        def step(self, llm: object, token_id: int, temp: float, top_k: int, top_p: float) -> npt.NDArray[np.float32]:
            del llm, token_id, temp, top_k, top_p
            return np.array([5.0, 0.0, 0.0], dtype=np.float32)

    model_dir = tmp_path / "gemma3-270m-it"
    _create_dummy_safetensors(model_dir)
    monkeypatch.setattr(model_module, "_core", CoreStub())
    mock_tokenizer.decode.return_value = ""

    prompt = "<start_of_turn>user\nhi\n<end_of_turn>\n<start_of_turn>model\n"
    model = SyncGemmaModel(GenerationConfig(model_path=model_dir, max_tokens=1))
    model.generate(prompt)

    encoded_prompt = mock_tokenizer.encode.call_args.args[0]
    assert encoded_prompt == prompt
