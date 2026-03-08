import json
import struct
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest

import mogemma.model as model_module
from mogemma import EmbeddingConfig, EmbeddingModel
from mogemma.hub import HubManager


def _create_dummy_safetensors(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "model.safetensors").open("wb") as f:
        h = json.dumps({}).encode("utf-8")
        f.write(struct.pack("<Q", len(h)) + h)
    (model_dir / "tokenizer.model").touch()


@pytest.fixture
def dummy_model_path(tmp_path: Path) -> str:
    """Fixture for a dummy model path."""
    model_dir = tmp_path / "bert-base-uncased"
    _create_dummy_safetensors(model_dir)
    return str(model_dir)


@pytest.fixture
def mock_tokenizer() -> Iterator[MagicMock]:
    """Fixture to mock the internal tokenizer."""
    with patch("mogemma.model._Tokenizer") as mock:
        tokenizer = MagicMock()

        def _encode_batch(inputs: str | list[str], **_: object) -> list[MagicMock]:
            batch_size = len(inputs) if isinstance(inputs, list) else 1
            result = []
            for _i in range(batch_size):
                encoded = MagicMock()
                encoded.ids = [1, 2, 3]
                result.append(encoded)
            return result

        tokenizer.encode_batch.side_effect = _encode_batch
        mock.return_value = tokenizer
        yield tokenizer


@pytest.fixture
def mock_core(monkeypatch: pytest.MonkeyPatch) -> object:
    class CoreStub:
        def init_model(self, _: str) -> object:
            return object()

        def generate_embeddings(self, llm: object, tokens: list[list[int]]) -> npt.NDArray[np.float32]:
            del llm
            return np.ones((len(tokens), 768), dtype=np.float32)

    core_stub = CoreStub()
    monkeypatch.setattr(model_module, "_core", core_stub)
    return core_stub


def test_embedding_model_string_config(dummy_model_path: str, mock_tokenizer: MagicMock, mock_core: object) -> None:
    """EmbeddingModel(str) should create a config with that model path."""
    model = EmbeddingModel(dummy_model_path)
    assert str(model.config.model_path) == dummy_model_path


def test_embedding_config(dummy_model_path: str) -> None:
    """Test embedding configuration initialization."""
    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    assert str(config.model_path) == dummy_model_path
    assert config.device == "cpu"


def test_embedding_model_init(dummy_model_path: str, mock_tokenizer: MagicMock, mock_core: object) -> None:
    """Test embedding model initialization."""
    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)
    assert model is not None


def test_embedding_model_init_uses_hub_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_tokenizer: MagicMock, mock_core: object
) -> None:
    """Embedding init should resolve through HubManager for HF-style IDs."""
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

    config = EmbeddingConfig(model_path="google/gemma-3-4b-it")
    model = EmbeddingModel(config)

    assert model is not None
    assert called == [("google/gemma-3-4b-it", True, True)]


def test_embedding_model_init_forwards_cache_path_to_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_tokenizer: MagicMock, mock_core: object
) -> None:
    """Embedding init should pass config.cache_path through to HubManager."""
    downloaded = tmp_path / "cached-model"
    downloaded.mkdir()
    _create_dummy_safetensors(downloaded)
    custom_cache = tmp_path / "custom-cache"
    observed_cache_paths: list[Path | None] = []

    class FakeHubManager:
        def __init__(self, cache_path: str | Path | None = None) -> None:
            observed_cache_paths.append(None if cache_path is None else Path(cache_path))

        def resolve_model(self, model_id: str, *, download_if_missing: bool, strict: bool, **_: object) -> Path:
            del model_id, download_if_missing, strict
            return downloaded

    monkeypatch.setattr(model_module, "HubManager", FakeHubManager)

    model = EmbeddingModel(EmbeddingConfig(model_path="google/gemma-3-4b-it", cache_path=custom_cache))
    assert model is not None
    assert observed_cache_paths == [custom_cache]


def test_embedding_model_init_rejects_unknown_local_path() -> None:
    config = EmbeddingConfig(model_path="bert-base-uncased-missing")

    with pytest.raises(HubManager.ModelNotFoundError, match="not found in the public gemma-data bucket"):
        EmbeddingModel(config)


def test_embed_single_string(dummy_model_path: str, mock_tokenizer: MagicMock, mock_core: object) -> None:
    """Test embedding a single string."""
    del mock_core
    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)
    embeddings = model.embed("Hello world")

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (1, 768)
    assert embeddings.dtype == np.float32


def test_embed_list_of_strings(dummy_model_path: str, mock_tokenizer: MagicMock, mock_core: object) -> None:
    """Test embedding a list of strings."""
    del mock_core
    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)
    texts = ["Hello", "Mojo runs Gemma inference."]
    embeddings = model.embed(texts)

    assert embeddings.shape == (2, 768)


def test_embed_raises_when_backend_returns_wrong_row_count(
    dummy_model_path: str, mock_tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure backend row-count mismatches are not silently accepted."""

    class CoreStub:
        def init_model(self, _: str) -> object:
            return object()

        def generate_embeddings(self, llm: object, tokens: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
            del llm, tokens
            return np.zeros((1, 768), dtype=np.float32)

    monkeypatch.setattr(model_module, "_core", CoreStub())

    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)

    with pytest.raises(ValueError, match="embedding rows"):
        model.embed(["first", "second"])


def test_embed_tokens_uses_mojo_without_tokenizer(dummy_model_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure pre-tokenized embedding path does not require transformers."""

    class CoreStub:
        def init_model(self, _: str) -> object:
            return object()

        def generate_embeddings(self, llm: object, tokens: list[list[int]]) -> npt.NDArray[np.float32]:
            del llm
            return np.ones((len(tokens), 768), dtype=np.float32)

    monkeypatch.setattr(model_module, "_core", CoreStub())
    monkeypatch.setattr(model_module, "SENTENCEPIECE_INSTALLED", False)

    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)
    embeddings = model.embed_tokens(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))

    assert embeddings.shape == (2, 768)
    assert embeddings.dtype == np.float32


def test_embed_rejects_empty_text_input(dummy_model_path: str, mock_tokenizer: MagicMock, mock_core: object) -> None:
    del mock_core
    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)

    with pytest.raises(ValueError, match="at least one value"):
        model.embed([])


def test_embed_tokens_rejects_empty_row(
    dummy_model_path: str, mock_core: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    del mock_core
    del monkeypatch
    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)

    with pytest.raises(ValueError, match="at least one row"):
        model.embed_tokens(np.empty((0, 1), dtype=np.int32))


def test_embed_text_requires_tokenizer_when_tokenizers_missing(
    dummy_model_path: str, monkeypatch: pytest.MonkeyPatch, mock_core: object
) -> None:
    """Ensure text embedding reports clear requirement when tokenizers is absent."""
    monkeypatch.setattr(model_module, "SENTENCEPIECE_INSTALLED", False)

    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)

    with pytest.raises(ModuleNotFoundError, match="embed_tokens"):
        model.embed("hello")


def test_embed_requires_mojo_core(
    dummy_model_path: str, mock_tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure embedding never falls back to non-Mojo output when core is missing."""
    monkeypatch.setattr(model_module, "_core", None)

    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    with pytest.raises(RuntimeError, match="Mojo core is unavailable"):
        EmbeddingModel(config)


def test_embedding_init_raises_on_core_init_failure(
    dummy_model_path: str, mock_tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    class CoreInitFailure:
        def init_model(self, _: str) -> object:
            msg = "checkpoint missing or unreadable"
            raise RuntimeError(msg)

        def generate_embeddings(self, llm: object, tokens: list[list[int]]) -> npt.NDArray[np.float32]:
            del llm, tokens
            return np.zeros((1, 768), dtype=np.float32)

    monkeypatch.setattr(model_module, "_core", CoreInitFailure())
    config = EmbeddingConfig(model_path=Path(dummy_model_path))

    with pytest.raises(RuntimeError, match="embedding model failed to initialize"):
        EmbeddingModel(config)


def test_embedding_model_rejects_unknown_backend(
    dummy_model_path: str, mock_tokenizer: MagicMock, mock_core: object
) -> None:
    del mock_tokenizer
    del mock_core
    config = EmbeddingConfig(model_path=Path(dummy_model_path), device="banana")

    with pytest.raises(ValueError, match="Unsupported backend"):
        EmbeddingModel(config)


def test_embedding_routes_through_backend_adapter(
    dummy_model_path: str, mock_tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    class CoreTrap:
        def init_model(self, _: str) -> object:
            return object()

        def generate_embeddings(self, llm: object, tokens: list[list[int]]) -> npt.NDArray[np.float32]:
            del llm, tokens
            msg = "EmbeddingModel should not call _core.generate_embeddings directly"
            raise AssertionError(msg)

    class BackendStub:
        backend_id = "test_backend"

        def __init__(self) -> None:
            self.embed_calls = 0

        def init_model(self, metadata: dict[str, tuple[int, tuple[int, ...]]]) -> object:
            del metadata
            return object()

        def generate_embeddings(self, llm: object, tokens: list[list[int]]) -> npt.NDArray[np.float32]:
            del llm
            self.embed_calls += 1
            return np.ones((len(tokens), 768), dtype=np.float32)

    backend = BackendStub()
    monkeypatch.setattr(model_module, "_core", CoreTrap())
    monkeypatch.setattr(model_module, "_resolve_embedding_backend", lambda *_args, **_kwargs: backend, raising=False)

    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)
    embeddings = model.embed_tokens([[1, 2, 3]])

    assert embeddings.shape == (1, 768)
    assert backend.embed_calls == 1


def test_embedding_model_defaults_to_cpu_backend(
    dummy_model_path: str, mock_tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BackendStub:
        backend_id = "cpu"

        def init_model(self, metadata: dict[str, tuple[int, tuple[int, ...], str]]) -> object:
            del metadata
            return object()

        def generate_embeddings(self, llm: object, tokens: list[list[int]]) -> npt.NDArray[np.float32]:
            del llm, tokens
            return np.ones((1, 768), dtype=np.float32)

    seen_devices: list[str] = []
    monkeypatch.setattr(model_module, "_core", object())
    monkeypatch.setattr(
        model_module,
        "_resolve_embedding_backend",
        lambda device: (seen_devices.append(device), BackendStub())[1],
        raising=False,
    )

    model = EmbeddingModel(EmbeddingConfig(model_path=Path(dummy_model_path), device="cpu"))

    assert model._backend_id == "cpu"
    assert seen_devices == ["cpu"]


def test_embedding_model_uses_normalized_backend_descriptor(
    dummy_model_path: str, mock_tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BackendStub:
        backend_id = "cpu"

        def init_model(self, metadata: dict[str, tuple[int, tuple[int, ...], str]]) -> object:
            del metadata
            return object()

        def generate_embeddings(self, llm: object, tokens: list[list[int]]) -> npt.NDArray[np.float32]:
            del llm, tokens
            return np.ones((1, 768), dtype=np.float32)

    seen: dict[str, object] = {}

    def fake_resolve_device_selection(device: str) -> object:
        seen["request"] = device
        return model_module.DeviceSelection(
            requested=device,
            backend="cpu",
            device_kind="cpu",
            device_index=None,
            strict=False,
            availability_source="override",
        )

    def fake_resolve_backend(device: str) -> BackendStub:
        seen["backend_device"] = device
        return BackendStub()

    monkeypatch.setattr(model_module, "_core", object())
    monkeypatch.setattr(model_module, "resolve_device_selection", fake_resolve_device_selection)
    monkeypatch.setattr(model_module, "_resolve_embedding_backend", fake_resolve_backend, raising=False)

    model = EmbeddingModel(EmbeddingConfig(model_path=Path(dummy_model_path), device="cpu"))

    assert seen["request"] == "cpu"
    assert seen["backend_device"] == "cpu"
    assert model._device_selection.effective_backend_id == "cpu"


def test_embedding_model_caches_device_selection_in_runtime_state(
    dummy_model_path: str, mock_tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BackendStub:
        backend_id = "cpu"

        def init_model(self, metadata: dict[str, tuple[int, tuple[int, ...], str]]) -> object:
            del metadata
            return {}

        def generate_embeddings(self, llm: object, tokens: list[list[int]]) -> npt.NDArray[np.float32]:
            del llm, tokens
            return np.ones((1, 768), dtype=np.float32)

    monkeypatch.setattr(model_module, "_core", object())
    monkeypatch.setattr(
        model_module,
        "resolve_device_selection",
        lambda device: model_module.DeviceSelection(
            requested=device,
            backend="cpu",
            device_kind="cpu",
            device_index=None,
            strict=False,
            availability_source="override",
        ),
    )
    monkeypatch.setattr(
        model_module, "_resolve_embedding_backend", lambda *_args, **_kwargs: BackendStub(), raising=False
    )

    model = EmbeddingModel(EmbeddingConfig(model_path=Path(dummy_model_path), device="cpu"))

    assert isinstance(model._llm, dict)
    assert model._llm["device_selection"] == {
        "requested": "cpu",
        "backend": "cpu",
        "device_kind": "cpu",
        "device_index": None,
        "strict": False,
        "availability_source": "override",
    }


def test_embedding_model_prefers_init_model_with_options_when_available(
    dummy_model_path: str, mock_tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    class OptionsCore:
        def __init__(self) -> None:
            self.seen_overrides: dict[str, int | float] | None = None
            self.seen_device_selection: dict[str, object] | None = None

        def init_model(self, _: object) -> object:
            msg = "legacy init_model path should not be used"
            raise AssertionError(msg)

        def init_model_with_options(
            self,
            metadata: dict[str, tuple[int, tuple[int, ...], str]],
            architecture_overrides: dict[str, int | float],
            device_selection: dict[str, object],
        ) -> object:
            del metadata
            self.seen_overrides = architecture_overrides
            self.seen_device_selection = device_selection
            return {}

        def generate_embeddings(self, llm: object, tokens: list[list[int]]) -> npt.NDArray[np.float32]:
            del llm, tokens
            return np.ones((1, 768), dtype=np.float32)

    core = OptionsCore()
    monkeypatch.setattr(model_module, "_core", core)

    model = EmbeddingModel(EmbeddingConfig(model_path=Path(dummy_model_path), device="cpu"))

    assert model is not None
    assert core.seen_overrides == {}
    assert core.seen_device_selection == {
        "requested": "cpu",
        "backend": "cpu",
        "device_kind": "cpu",
        "device_index": None,
        "strict": False,
        "availability_source": "cpu-default",
    }
    assert isinstance(model._llm, dict)
    assert model._llm["device_selection"] == core.seen_device_selection


def test_embed_token_array_enforces_rectangular_shape(
    dummy_model_path: str, mock_tokenizer: MagicMock, mock_core: object
) -> None:
    del mock_tokenizer
    del mock_core
    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)

    with pytest.raises(ValueError, match="all token sequences in a batch must have the exact same length"):
        model.embed_tokens([[1, 2, 3], [1, 2]])


def test_embed_pads_heterogeneous_sequences(
    dummy_model_path: str, mock_tokenizer: MagicMock, mock_core: object
) -> None:
    """Test that embedding multiple strings of different lengths results in equal-length token arrays.

    This ensures that token arrays passed to the backend are rectangular.
    """
    del mock_core

    # Setup mock tokenizer to return heterogeneous encodings
    class MockEncoding:
        def __init__(self, ids: list[int]) -> None:
            self.ids = ids

    mock_tokenizer.encode_batch.return_value = [
        MockEncoding([1, 2, 3, 0, 0]),
        MockEncoding([4, 5, 0, 0, 0]),
        MockEncoding([6, 7, 8, 9, 10]),
    ]

    config = EmbeddingConfig(model_path=Path(dummy_model_path))
    model = EmbeddingModel(config)

    # Text input that would normally be heterogeneous
    texts = ["Long text here", "Short", "Very long text goes here indeed"]

    # embed() should call tokenizer.enable_padding() which ensures equal lengths
    # The actual padding happens in the mock (we simulate it returning equal lengths).
    # We verify that model.embed does not throw the ValueError about rectangular arrays.
    embeddings = model.embed(texts)

    assert embeddings.shape == (3, 768)
    mock_tokenizer.enable_padding.assert_called_once()
