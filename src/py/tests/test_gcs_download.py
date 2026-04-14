"""Tests for the GCS download backend in hub.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mogemma.hub import HubManager


class TestMakeGCSStore:
    """Tests for GCSStore creation against the public gemma-data bucket."""

    def test_creates_store(self) -> None:
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        store = hub._make_gcs_store()
        assert store is not None

    def test_store_is_gcs_store_instance(self) -> None:
        from obstore.store import GCSStore

        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        store = hub._make_gcs_store()
        assert isinstance(store, GCSStore)


class TestGCSPathHelpers:
    """Tests for GCS path helpers."""

    def test_checkpoint_prefix_trailing_slash(self) -> None:
        assert HubManager._gcs_checkpoint_prefix("gemma4-e2b-it") == "checkpoints/gemma4-e2b-it/"

    def test_checkpoint_prefix_lowercased_input(self) -> None:
        assert HubManager._gcs_checkpoint_prefix("gemma4-26b-a4b-pt") == "checkpoints/gemma4-26b-a4b-pt/"

    def test_tokenizer_path_is_shared(self) -> None:
        assert HubManager._gcs_tokenizer_path() == "tokenizers/tokenizer_gemma4.model"


class TestCleanModelIdLowercase:
    """Tests for lowercase normalization of cleaned model ids for GCS."""

    def test_google_prefix_stripped_and_lowercased(self) -> None:
        assert HubManager._clean_model_id("google/gemma-4-26B-A4B-it") == "gemma4-26b-a4b-it"

    def test_dash_removed_and_lowercased(self) -> None:
        assert HubManager._clean_model_id("gemma-4-E2B-pt") == "gemma4-e2b-pt"

    def test_already_clean_lowercased(self) -> None:
        assert HubManager._clean_model_id("gemma4-31b-it") == "gemma4-31b-it"

    def test_mixed_case_e4b(self) -> None:
        assert HubManager._clean_model_id("google/gemma-4-E4B-IT") == "gemma4-e4b-it"


class TestListRemoteFiles:
    """Tests for enumerating checkpoint file listings from GCS."""

    def test_list_returns_relative_paths(self) -> None:
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        pages = [
            [
                {"path": "checkpoints/gemma4-e2b-it/_METADATA", "size": 1},
                {"path": "checkpoints/gemma4-e2b-it/manifest.ocdbt", "size": 2},
            ],
            [{"path": "checkpoints/gemma4-e2b-it/ocdbt.process_0/d/abc", "size": 3}],
        ]
        with patch("mogemma.hub.obs.list", return_value=iter(pages)):
            files = hub._list_remote_files(MagicMock(), "checkpoints/gemma4-e2b-it/")
        assert files == ["_METADATA", "manifest.ocdbt", "ocdbt.process_0/d/abc"]

    def test_list_filters_folder_markers(self) -> None:
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        pages = [
            [
                {"path": "checkpoints/gemma4-e2b-it/d_$folder$", "size": 0},
                {"path": "checkpoints/gemma4-e2b-it/descriptor_$folder$", "size": 0},
                {"path": "checkpoints/gemma4-e2b-it/_METADATA", "size": 1},
            ]
        ]
        with patch("mogemma.hub.obs.list", return_value=iter(pages)):
            files = hub._list_remote_files(MagicMock(), "checkpoints/gemma4-e2b-it/")
        assert "_METADATA" in files
        assert all("_$folder$" not in f for f in files)

    def test_list_skips_prefix_itself(self) -> None:
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        pages = [
            [
                # Some stores return the "directory" entry itself with an empty relative name.
                {"path": "checkpoints/gemma4-e2b-it/", "size": 0},
                {"path": "checkpoints/gemma4-e2b-it/_METADATA", "size": 1},
            ]
        ]
        with patch("mogemma.hub.obs.list", return_value=iter(pages)):
            files = hub._list_remote_files(MagicMock(), "checkpoints/gemma4-e2b-it/")
        assert files == ["_METADATA"]

    @pytest.mark.asyncio
    async def test_list_async_returns_relative_paths(self) -> None:
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
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
            files = await hub._list_remote_files_async(MagicMock(), "checkpoints/gemma4-e2b-it/")
        assert files == ["_METADATA", "manifest.ocdbt"]


class TestOrbaxDetection:
    """Tests for _has_model_files detecting Orbax in addition to safetensors."""

    def test_orbax_dir_detected(self, tmp_path: Path) -> None:
        (tmp_path / "ocdbt.process_0").mkdir()
        (tmp_path / "manifest.ocdbt").write_bytes(b"")
        assert HubManager._has_model_files(tmp_path) is True

    def test_orbax_requires_both_files(self, tmp_path: Path) -> None:
        # Only directory, no manifest → not detected.
        (tmp_path / "ocdbt.process_0").mkdir()
        assert HubManager._has_model_files(tmp_path) is False

    def test_safetensors_still_detected(self, tmp_path: Path) -> None:
        (tmp_path / "model.safetensors").write_bytes(b"")
        assert HubManager._has_model_files(tmp_path) is True


class TestDeletedHFCode:
    """Ensure HF-specific methods no longer exist."""

    def test_make_hf_store_removed(self) -> None:
        assert not hasattr(HubManager, "_make_hf_store")

    def test_get_hf_token_removed(self) -> None:
        assert not hasattr(HubManager, "_get_hf_token")

    def test_hf_resolve_url_removed(self) -> None:
        assert not hasattr(HubManager, "_hf_resolve_url")

    def test_parse_shard_filenames_removed(self) -> None:
        assert not hasattr(HubManager, "_parse_shard_filenames")

    def test_fetch_index_json_removed(self) -> None:
        assert not hasattr(HubManager, "_fetch_index_json")

    def test_hf_download_error_removed(self) -> None:
        assert not hasattr(HubManager, "HFDownloadError")

    def test_gcs_download_error_present(self) -> None:
        assert hasattr(HubManager, "GCSDownloadError")
        assert issubclass(HubManager.GCSDownloadError, ConnectionError)


class TestGetTokenizerPathLocalName:
    """_get_tokenizer_path returns the local cache filename."""

    def test_gemma4_local_name(self) -> None:
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        assert hub._get_tokenizer_path("gemma4-e2b-it") == "tokenizer.model"

    def test_non_gemma4_returns_none(self) -> None:
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        assert hub._get_tokenizer_path("llama-7b") is None
