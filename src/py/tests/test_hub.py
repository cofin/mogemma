"""Tests for hub tokenizer path resolution and model file detection."""

from __future__ import annotations

from pathlib import Path

from mogemma.hub import HubManager


class TestGetTokenizerPath:
    def _get_path(self, clean_id: str) -> str | None:
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        return hub._get_tokenizer_path(clean_id)

    def test_gemma4_model(self) -> None:
        assert self._get_path("gemma4-31b-it") == "tokenizer.model"

    def test_gemma4_dense_model(self) -> None:
        assert self._get_path("gemma4-e2b-it") == "tokenizer.model"

    def test_unknown_model_returns_none(self) -> None:
        assert self._get_path("llama-7b") is None

    def test_no_gemma2_support(self) -> None:
        """Old gemma2 models should not resolve a tokenizer path."""
        assert self._get_path("gemma2-2b-it") is None

    def test_no_gemma3_support(self) -> None:
        """Old gemma3 models should not resolve a tokenizer path."""
        assert self._get_path("gemma3-27b-it") is None


class TestHasModelFiles:
    def test_empty_dir_returns_false(self, tmp_path: Path) -> None:
        assert HubManager._has_model_files(tmp_path) is False

    def test_safetensors_dir_returns_true(self, tmp_path: Path) -> None:
        (tmp_path / "model.safetensors").write_bytes(b"")
        assert HubManager._has_model_files(tmp_path) is True


class TestCleanModelId:
    def test_strips_google_prefix(self) -> None:
        # Cleaned IDs are lowercased to match the GCS checkpoint path convention.
        assert HubManager._clean_model_id("google/gemma-4-26B-A4B-it") == "gemma4-26b-a4b-it"

    def test_dash_to_no_dash(self) -> None:
        assert HubManager._clean_model_id("gemma-4-26B-A4B-it") == "gemma4-26b-a4b-it"

    def test_already_clean(self) -> None:
        assert HubManager._clean_model_id("gemma4-26b-a4b-it") == "gemma4-26b-a4b-it"
