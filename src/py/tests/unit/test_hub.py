"""Unit tests for hub catalog, path helpers, downloads, and local finalization."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mogemma.hub import KNOWN_GCS_MODELS, HubManager


def _hub(cache_path: Path = Path("/tmp/test-cache")) -> HubManager:
    return HubManager(cache_path=cache_path)


def test_known_gcs_models_catalog_shape() -> None:
    assert isinstance(KNOWN_GCS_MODELS, frozenset)
    assert len(KNOWN_GCS_MODELS) >= 1
    assert all(isinstance(model_id, str) for model_id in KNOWN_GCS_MODELS)
    assert all(model_id.startswith("google/") for model_id in KNOWN_GCS_MODELS)


def test_known_gcs_models_include_current_public_catalog_entries() -> None:
    expected = {"google/gemma-4-E2B-it", "google/gemma-4-E4B-it", "google/gemma-4-26B-A4B-it"}
    assert expected.issubset(KNOWN_GCS_MODELS)


def test_known_gcs_models_exclude_12b_without_live_gcs_proof() -> None:
    assert "google/gemma-4-12B-it" not in KNOWN_GCS_MODELS


def test_make_gcs_store_returns_obstore_gcs_store() -> None:
    from obstore.store import GCSStore

    assert isinstance(_hub()._make_gcs_store(), GCSStore)


@pytest.mark.parametrize(
    ("model_id", "cleaned"),
    [
        ("google/gemma-4-26B-A4B-it", "gemma4-26b-a4b-it"),
        ("gemma-4-26B-A4B-it", "gemma4-26b-a4b-it"),
        ("google/gemma-4-E2B-IT", "gemma4-e2b-it"),
        ("gemma4-31b-it", "gemma4-31b-it"),
    ],
)
def test_clean_model_id_normalizes_to_gcs_checkpoint_name(model_id: str, cleaned: str) -> None:
    assert HubManager._clean_model_id(model_id) == cleaned


def test_gcs_checkpoint_and_tokenizer_paths() -> None:
    assert HubManager._gcs_checkpoint_prefix("gemma4-e2b-it") == "checkpoints/gemma4-e2b-it/"
    assert HubManager._gcs_tokenizer_path() == "tokenizers/tokenizer_gemma4.model"


@pytest.mark.parametrize(
    ("clean_id", "tokenizer_name"),
    [
        ("gemma4-31b-it", "tokenizer.model"),
        ("gemma4-e2b-it", "tokenizer.model"),
        ("gemma4-e4b-it", "tokenizer.model"),
        ("llama-7b", None),
    ],
)
def test_get_tokenizer_path_for_clean_model_id(clean_id: str, tokenizer_name: str | None) -> None:
    assert _hub()._get_tokenizer_path(clean_id) == tokenizer_name


def test_resolve_tokenizer_prefers_explicit_path(tmp_path: Path) -> None:
    explicit = tmp_path / "custom" / "my_tokenizer.model"
    explicit.parent.mkdir(parents=True)
    explicit.write_bytes(b"custom-tokenizer")

    assert _hub(tmp_path).resolve_tokenizer(tmp_path, tokenizer_path=explicit) == explicit


def test_resolve_tokenizer_uses_local_model_tokenizer(tmp_path: Path) -> None:
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"fake-tokenizer")

    assert _hub(tmp_path).resolve_tokenizer(tmp_path) == tokenizer


@pytest.mark.parametrize("tokenizer_path", [None, Path("missing.model")])
def test_resolve_tokenizer_raises_when_required_file_is_missing(tokenizer_path: Path | None, tmp_path: Path) -> None:
    resolved_path = None if tokenizer_path is None else tmp_path / tokenizer_path
    with pytest.raises(FileNotFoundError, match="tokenizer"):
        _hub(tmp_path).resolve_tokenizer(tmp_path, tokenizer_path=resolved_path)


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "gemma4", "num_hidden_layers": 60, "hidden_size": 4096},
        {"model_type": "gemma4_text", "num_hidden_layers": 60, "hidden_size": 4096},
    ],
)
def test_validate_config_json_accepts_gemma4_configs(config: dict[str, object]) -> None:
    HubManager.validate_config_json(config)


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"num_hidden_layers": 60}, "model_type"),
        ({"model_type": "llama", "num_hidden_layers": 32}, "Gemma 4"),
        ({"model_type": "gemma4"}, "num_hidden_layers"),
    ],
)
def test_validate_config_json_rejects_invalid_configs(config: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        HubManager.validate_config_json(config)


def test_list_remote_files_returns_relative_paths() -> None:
    pages = [
        [
            {"path": "checkpoints/gemma4-e2b-it/_METADATA", "size": 1},
            {"path": "checkpoints/gemma4-e2b-it/manifest.ocdbt", "size": 2},
        ],
        [{"path": "checkpoints/gemma4-e2b-it/ocdbt.process_0/d/abc", "size": 3}],
    ]
    with patch("mogemma.hub.obs.list", return_value=iter(pages)):
        files = _hub()._list_remote_files(MagicMock(), "checkpoints/gemma4-e2b-it/")
    assert files == ["_METADATA", "manifest.ocdbt", "ocdbt.process_0/d/abc"]


def test_list_remote_files_filters_folder_markers_and_prefix_entry() -> None:
    pages = [
        [
            {"path": "checkpoints/gemma4-e2b-it/", "size": 0},
            {"path": "checkpoints/gemma4-e2b-it/d_$folder$", "size": 0},
            {"path": "checkpoints/gemma4-e2b-it/_METADATA", "size": 1},
        ]
    ]
    with patch("mogemma.hub.obs.list", return_value=iter(pages)):
        assert _hub()._list_remote_files(MagicMock(), "checkpoints/gemma4-e2b-it/") == ["_METADATA"]


@pytest.mark.asyncio
async def test_list_remote_files_async_returns_relative_paths() -> None:
    pages = [
        [
            {"path": "checkpoints/gemma4-e2b-it/_METADATA", "size": 1},
            {"path": "checkpoints/gemma4-e2b-it/manifest.ocdbt", "size": 2},
        ]
    ]

    class _AsyncIter:
        def __init__(self, items: list[list[dict[str, object]]]) -> None:
            self._items = iter(items)

        def __aiter__(self) -> _AsyncIter:
            return self

        async def __anext__(self) -> list[dict[str, object]]:
            try:
                return next(self._items)
            except StopIteration:
                raise StopAsyncIteration from None

    with patch("mogemma.hub.obs.list", return_value=_AsyncIter(pages)):
        files = await _hub()._list_remote_files_async(MagicMock(), "checkpoints/gemma4-e2b-it/")
    assert files == ["_METADATA", "manifest.ocdbt"]


@pytest.mark.parametrize(
    ("files", "expected_result"),
    [
        ({}, False),
        ({"model.safetensors": b""}, True),
        ({"ocdbt.process_0/manifest": b"", "manifest.ocdbt": b""}, True),
        ({"ocdbt.process_0/manifest": b""}, False),
    ],
)
def test_has_model_files_detects_supported_local_layouts(
    files: dict[str, bytes], expected_result: object, tmp_path: Path
) -> None:
    for relative_path, content in files.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    assert HubManager._has_model_files(tmp_path) is expected_result


def test_download_file_streams_result_chunks(tmp_path: Path) -> None:
    class Result:
        def stream(self) -> list[bytes]:
            return [b"abc", b"def"]

        def bytes(self) -> bytes:
            msg = "download should use stream()"
            raise AssertionError(msg)

    with (
        patch("mogemma.hub.obs.get", return_value=Result()),
        patch.object(HubManager, "_write_file", side_effect=AssertionError("_write_file should not be used")),
    ):
        _hub(tmp_path / "cache")._download_file(object(), "remote", tmp_path / "model.bin")  # type: ignore[arg-type]

    assert (tmp_path / "model.bin").read_bytes() == b"abcdef"


@pytest.mark.asyncio
async def test_download_file_async_streams_result_chunks(tmp_path: Path) -> None:
    class Chunks:
        def __init__(self, chunks: list[bytes]) -> None:
            self._chunks = iter(chunks)

        def __iter__(self) -> Chunks:
            return self

        def __next__(self) -> bytes:
            try:
                return next(self._chunks)
            except StopIteration:
                raise StopIteration from None

        def __aiter__(self) -> Chunks:
            return self

        async def __anext__(self) -> bytes:
            try:
                return next(self)
            except StopIteration:
                raise StopAsyncIteration from None

    class Result:
        def stream(self) -> Chunks:
            return Chunks([b"abc", b"def"])

        async def bytes_async(self) -> bytes:
            msg = "download should use stream()"
            raise AssertionError(msg)

    with patch("mogemma.hub.obs.get_async", new=AsyncMock(return_value=Result())):
        await _hub(tmp_path / "cache")._download_file_async(object(), "remote", tmp_path / "model.bin")  # type: ignore[arg-type]

    assert (tmp_path / "model.bin").read_bytes() == b"abcdef"


def _populate_orbax(dir_path: Path) -> None:
    """Lay down a minimal Orbax/OCDBT-shaped directory."""
    (dir_path / "ocdbt.process_0").mkdir(parents=True)
    (dir_path / "ocdbt.process_0" / "manifest").write_bytes(b"stub")
    (dir_path / "manifest.ocdbt").write_bytes(b"stub")
    (dir_path / "_METADATA").write_bytes(b"stub")
    (dir_path / "_CHECKPOINT_METADATA").write_bytes(b"stub")
    (dir_path / "descriptor").mkdir()
    (dir_path / "descriptor" / "x").write_bytes(b"stub")
    (dir_path / "d").mkdir()
    (dir_path / "d" / "y").write_bytes(b"stub")
    (dir_path / "commit_success.txt").write_bytes(b"stub")
    (dir_path / "config.json").write_bytes(b"{}")
    (dir_path / "tokenizer.model").write_bytes(b"stub")


def test_finalize_download_converts_orbax_and_cleans_artifacts(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    staging = cache_root / ".staging"
    staging.mkdir()
    _populate_orbax(staging)
    local_dir = cache_root / "final"

    def fake_convert(path: Path) -> list[Path]:
        out = path / "model.safetensors"
        out.write_bytes(b"fake")
        return [out]

    with patch("mogemma.convert.convert_orbax_to_safetensors", side_effect=fake_convert):
        result = _hub(cache_root)._finalize_download("gemma4-e4b", local_dir, staging, tokenizer_required=True)

    assert result == local_dir
    assert (local_dir / "model.safetensors").exists()
    assert (local_dir / "config.json").exists()
    assert (local_dir / "tokenizer.model").exists()
    assert not (local_dir / "ocdbt.process_0").exists()
    assert not (local_dir / "manifest.ocdbt").exists()
    assert not (local_dir / "_METADATA").exists()
    assert not (local_dir / "_CHECKPOINT_METADATA").exists()
    assert not (local_dir / "descriptor").exists()
    assert not (local_dir / "d").exists()
    assert not (local_dir / "commit_success.txt").exists()


def test_finalize_download_preserves_orbax_on_conversion_failure(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    staging = cache_root / ".staging"
    staging.mkdir()
    _populate_orbax(staging)
    local_dir = cache_root / "final"

    def boom(_path: Path) -> list[Path]:
        msg = "simulated conversion failure"
        raise RuntimeError(msg)

    with (
        patch("mogemma.convert.convert_orbax_to_safetensors", side_effect=boom),
        pytest.raises(RuntimeError, match="simulated conversion failure"),
    ):
        _hub(cache_root)._finalize_download("gemma4-e4b", local_dir, staging, tokenizer_required=True)

    assert (local_dir / "ocdbt.process_0").is_dir()
    assert (local_dir / "manifest.ocdbt").exists()
    assert not (local_dir / "model.safetensors").exists()


def test_finalize_download_skips_conversion_when_safetensors_already_present(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    staging = cache_root / ".staging"
    staging.mkdir()
    (staging / "model.safetensors").write_bytes(b"pre-existing")
    (staging / "config.json").write_bytes(b"{}")
    (staging / "tokenizer.model").write_bytes(b"stub")
    local_dir = cache_root / "final"

    with patch("mogemma.convert.convert_orbax_to_safetensors") as mock_convert:
        _hub(cache_root)._finalize_download("gemma4-e4b", local_dir, staging, tokenizer_required=True)
        mock_convert.assert_not_called()

    assert (local_dir / "model.safetensors").read_bytes() == b"pre-existing"


def test_has_model_files_accepts_safetensors_when_orbax_artifacts_are_also_present(tmp_path: Path) -> None:
    (tmp_path / "ocdbt.process_0").mkdir()
    (tmp_path / "manifest.ocdbt").write_bytes(b"")
    (tmp_path / "model.safetensors").write_bytes(b"")

    assert HubManager._has_safetensors(tmp_path) is True
    assert HubManager._has_orbax(tmp_path) is True
    assert HubManager._has_model_files(tmp_path) is True


def test_12b_remote_download_is_not_overclaimed_without_gcs_objects(tmp_path: Path) -> None:
    hub = _hub(tmp_path)
    with (
        patch.object(HubManager, "_list_remote_files", return_value=[]),
        pytest.raises(HubManager.ModelNotFoundError, match="No objects found"),
    ):
        hub.resolve_model("google/gemma-4-12B-it", download_if_missing=True, strict=True)
