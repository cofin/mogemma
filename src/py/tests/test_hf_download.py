"""Tests for HuggingFace HTTP download backend in hub.py."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from mogemma.hub import HubManager


class TestMakeHFStore:
    """Tests for HTTPStore creation with HuggingFace base URL."""

    def test_creates_store_for_repo(self) -> None:
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        store = hub._make_hf_store("google/gemma-4-31B-it")
        assert store is not None

    def test_store_url_contains_repo_id(self) -> None:
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        store = hub._make_hf_store("google/gemma-4-31B-it")
        # HTTPStore stores the base URL
        assert "huggingface.co" in store.url

    def test_store_with_token(self) -> None:
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        store = hub._make_hf_store("google/gemma-4-31B-it", token="hf_test123")
        assert store is not None

    def test_store_without_token(self) -> None:
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        store = hub._make_hf_store("google/gemma-4-31B-it", token=None)
        assert store is not None


class TestHFTokenDetection:
    """Tests for HF_TOKEN environment variable handling."""

    def test_reads_hf_token_env(self) -> None:
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        with patch.dict(os.environ, {"HF_TOKEN": "hf_abc123"}):
            token = hub._get_hf_token()
            assert token == "hf_abc123"

    def test_returns_none_when_no_token(self) -> None:
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        with patch.dict(os.environ, {}, clear=True):
            # Remove HF_TOKEN if present
            os.environ.pop("HF_TOKEN", None)
            token = hub._get_hf_token()
            assert token is None


class TestListHFShardFiles:
    """Tests for shard file list parsing from model.safetensors.index.json."""

    def test_parses_weight_map_to_unique_filenames(self) -> None:
        index_json = {
            "metadata": {"total_size": 1000},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00003.safetensors",
                "model.layers.0.self_attn.k_proj.weight": "model-00002-of-00003.safetensors",
                "model.layers.1.mlp.up_proj.weight": "model-00003-of-00003.safetensors",
            },
        }
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        filenames = hub._parse_shard_filenames(index_json)
        assert sorted(filenames) == [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ]

    def test_raises_on_missing_weight_map(self) -> None:
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        with pytest.raises(ValueError, match="weight_map"):
            hub._parse_shard_filenames({"metadata": {}})

    def test_raises_on_empty_weight_map(self) -> None:
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        with pytest.raises(ValueError, match="weight_map"):
            hub._parse_shard_filenames({"weight_map": {}})


class TestDownloadHFFile:
    """Tests for single file download with size-check skip logic."""

    def test_skips_download_when_file_exists_with_correct_size(self, tmp_path: Path) -> None:
        dest = tmp_path / "model-00001.safetensors"
        dest.write_bytes(b"x" * 100)
        hub = HubManager(cache_path=tmp_path)
        # Should return True (skipped) when file exists with expected size
        skipped = hub._should_skip_download(dest, expected_size=100)
        assert skipped is True

    def test_does_not_skip_when_size_mismatch(self, tmp_path: Path) -> None:
        dest = tmp_path / "model-00001.safetensors"
        dest.write_bytes(b"x" * 50)
        hub = HubManager(cache_path=tmp_path)
        skipped = hub._should_skip_download(dest, expected_size=100)
        assert skipped is False

    def test_does_not_skip_when_file_missing(self, tmp_path: Path) -> None:
        dest = tmp_path / "model-00001.safetensors"
        hub = HubManager(cache_path=tmp_path)
        skipped = hub._should_skip_download(dest, expected_size=100)
        assert skipped is False

    def test_does_not_skip_when_no_expected_size(self, tmp_path: Path) -> None:
        dest = tmp_path / "model-00001.safetensors"
        dest.write_bytes(b"x" * 100)
        hub = HubManager(cache_path=tmp_path)
        skipped = hub._should_skip_download(dest, expected_size=None)
        assert skipped is False


class TestHFURLConstruction:
    """Tests for HuggingFace URL pattern construction."""

    def test_resolve_url(self) -> None:
        assert (
            HubManager._hf_resolve_url("google/gemma-4-31B-it", "config.json")
            == "https://huggingface.co/google/gemma-4-31B-it/resolve/main/config.json"
        )

    def test_resolve_url_shard(self) -> None:
        url = HubManager._hf_resolve_url("google/gemma-4-31B-it", "model-00001-of-00020.safetensors")
        assert url == "https://huggingface.co/google/gemma-4-31B-it/resolve/main/model-00001-of-00020.safetensors"


class TestDownloadSyncHF:
    """Tests for the rewritten download_sync using HF backend."""

    def test_raises_model_not_found_on_404(self, tmp_path: Path) -> None:
        """download_sync should raise ModelNotFoundError when index.json fetch fails."""
        hub = HubManager(cache_path=tmp_path)
        with pytest.raises(HubManager.ModelNotFoundError):
            # Non-existent model — will fail to fetch index.json
            hub.download_sync("google/nonexistent-model-xyz")


class TestDeletedGCSCode:
    """Verify GCS-specific code has been removed."""

    def test_no_gcs_store_attribute(self) -> None:
        """HubManager should not have _make_store returning GCSStore."""
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        # The old _make_store returned GCSStore. It should no longer exist
        # or should return an HTTPStore instead.
        assert not hasattr(hub, "_make_gcs_store")

    def test_no_gcs_download_error(self) -> None:
        """GCSDownloadError should be replaced with HFDownloadError."""
        assert hasattr(HubManager, "HFDownloadError")

    def test_error_messages_no_gcs_mention(self, tmp_path: Path) -> None:
        """Error messages should not mention GCS or Google Cloud Storage."""
        hub = HubManager(cache_path=tmp_path)
        with pytest.raises(HubManager.ModelNotFoundError) as exc_info:
            hub.download_sync("google/nonexistent-model-xyz")
        msg = str(exc_info.value)
        assert "GCS" not in msg
        assert "Google Cloud Storage" not in msg
