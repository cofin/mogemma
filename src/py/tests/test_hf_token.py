"""Tests for HF_TOKEN environment variable handling and Bearer header injection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

from mogemma.hub import HubManager

if TYPE_CHECKING:
    from obstore.store import HTTPStore


class TestHFTokenEnvVar:
    """Verify HF_TOKEN is detected and passed as Bearer header."""

    def test_token_from_env(self) -> None:
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        with patch.dict(os.environ, {"HF_TOKEN": "hf_secret_123"}):
            assert hub._get_hf_token() == "hf_secret_123"

    def test_no_token_returns_none(self) -> None:
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("HF_TOKEN", None)
            assert hub._get_hf_token() is None

    def test_empty_token_returns_empty(self) -> None:
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        with patch.dict(os.environ, {"HF_TOKEN": ""}):
            assert hub._get_hf_token() == ""

    def test_store_created_without_token_for_public_models(self) -> None:
        """Public models should work without HF_TOKEN."""
        store = HubManager._make_hf_store("google/gemma-4-26B-A4B-it", token=None)
        assert store is not None

    def test_store_created_with_token_for_gated_models(self) -> None:
        """Gated models need HF_TOKEN for Bearer auth."""
        store = HubManager._make_hf_store("google/gemma-4-26B-A4B-it", token="hf_test_token")
        assert store is not None


class TestDownloadSyncUsesToken:
    """Verify download_sync reads HF_TOKEN and passes it to the store."""

    def test_download_sync_reads_token(self, tmp_path: Path) -> None:
        """download_sync should call _get_hf_token() and use it."""
        hub = HubManager(cache_path=tmp_path)
        calls: list[str | None] = []
        original_make_hf_store = HubManager._make_hf_store

        @staticmethod  # type: ignore[misc]
        def tracking_make_store(repo_id: str, token: str | None = None) -> HTTPStore:
            calls.append(token)
            return original_make_hf_store(repo_id, token=token)

        with (
            patch.dict(os.environ, {"HF_TOKEN": "hf_tracked"}),
            patch.object(HubManager, "_make_hf_store", tracking_make_store),
        ):
            try:
                hub.download_sync("google/gemma-4-26B-A4B-it")
            except Exception:  # noqa: BLE001
                pass  # Will fail on network — we just want to verify token was passed

        assert calls == ["hf_tracked"]
