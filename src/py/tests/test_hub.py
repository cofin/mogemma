"""Tests for hub tokenizer path resolution and model file detection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from mogemma.hub import KNOWN_GCS_MODELS, HubManager


class TestKnownGCSModels:
    """The module-level catalog of models known to exist in gs://gemma-data.

    Kept as a hand-maintained frozenset because the bucket is public read-only
    and adding models is a manual ops step, not a runtime discovery. The live
    probe in test_hub_live.py verifies each entry still has objects.
    """

    def test_is_frozenset_of_strings(self) -> None:
        assert isinstance(KNOWN_GCS_MODELS, frozenset)
        assert all(isinstance(m, str) for m in KNOWN_GCS_MODELS)

    def test_nonempty(self) -> None:
        assert len(KNOWN_GCS_MODELS) >= 1

    def test_entries_are_google_prefixed_model_ids(self) -> None:
        for model_id in KNOWN_GCS_MODELS:
            assert model_id.startswith("google/"), f"Unexpected catalog entry: {model_id!r}"

    def test_contains_three_currently_available_models(self) -> None:
        """Matches the 2026-04-16 gs://gemma-data probe result."""
        expected = {"google/gemma-4-E2B-it", "google/gemma-4-E4B-it", "google/gemma-4-26B-A4B-it"}
        assert expected.issubset(KNOWN_GCS_MODELS)

    def test_no_stale_gemma3_entries(self) -> None:
        for model_id in KNOWN_GCS_MODELS:
            lowered = model_id.lower()
            assert "gemma3-" not in lowered, f"Stale Gemma 3 entry: {model_id!r}"
            assert "gemma3n-" not in lowered, f"Stale Gemma 3n entry: {model_id!r}"

    def test_no_missing_pretrained_variants(self) -> None:
        """Catalog must not lie about what's actually in the bucket.

        Pretrained E2B/E4B are not in ``gs://gemma-data`` as of 2026-04-16.
        When Google publishes them, add them and delete this guard.
        """
        for model_id in KNOWN_GCS_MODELS:
            assert model_id not in {"google/gemma-4-E2B", "google/gemma-4-E4B"}, (
                f"Pretrained variant {model_id!r} is not in gs://gemma-data; remove from catalog."
            )


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


class TestFinalizeDownloadOrbaxConversion:
    """`_finalize_download` converts Orbax layouts to safetensors and cleans up."""

    def test_converts_orbax_and_cleans_artifacts(self, tmp_path: Path) -> None:
        cache_root = tmp_path / "cache"
        cache_root.mkdir()
        hub = HubManager(cache_path=cache_root)
        staging = cache_root / ".staging"
        staging.mkdir()
        _populate_orbax(staging)
        local_dir = cache_root / "final"

        def fake_convert(path: Path) -> list[Path]:
            # Simulate conversion by writing a safetensors artifact into the dir.
            out = path / "model.safetensors"
            out.write_bytes(b"fake")
            return [out]

        with patch("mogemma.convert.convert_orbax_to_safetensors", side_effect=fake_convert):
            result = hub._finalize_download("gemma4-e4b", local_dir, staging, tokenizer_required=True)

        assert result == local_dir
        assert (local_dir / "model.safetensors").exists()
        assert (local_dir / "config.json").exists()
        assert (local_dir / "tokenizer.model").exists()
        # Orbax artifacts removed after successful conversion.
        assert not (local_dir / "ocdbt.process_0").exists()
        assert not (local_dir / "manifest.ocdbt").exists()
        assert not (local_dir / "_METADATA").exists()
        assert not (local_dir / "_CHECKPOINT_METADATA").exists()
        assert not (local_dir / "descriptor").exists()
        assert not (local_dir / "d").exists()
        assert not (local_dir / "commit_success.txt").exists()

    def test_preserves_orbax_on_conversion_failure(self, tmp_path: Path) -> None:
        cache_root = tmp_path / "cache"
        cache_root.mkdir()
        hub = HubManager(cache_path=cache_root)
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
            hub._finalize_download("gemma4-e4b", local_dir, staging, tokenizer_required=True)

        # On failure the Orbax layout must be preserved for retry.
        assert (local_dir / "ocdbt.process_0").is_dir()
        assert (local_dir / "manifest.ocdbt").exists()
        assert not (local_dir / "model.safetensors").exists()

    def test_skips_conversion_when_safetensors_already_present(self, tmp_path: Path) -> None:
        cache_root = tmp_path / "cache"
        cache_root.mkdir()
        hub = HubManager(cache_path=cache_root)
        staging = cache_root / ".staging"
        staging.mkdir()
        (staging / "model.safetensors").write_bytes(b"pre-existing")
        (staging / "config.json").write_bytes(b"{}")
        (staging / "tokenizer.model").write_bytes(b"stub")
        local_dir = cache_root / "final"

        with patch("mogemma.convert.convert_orbax_to_safetensors") as mock_convert:
            hub._finalize_download("gemma4-e4b", local_dir, staging, tokenizer_required=True)
            mock_convert.assert_not_called()

        assert (local_dir / "model.safetensors").read_bytes() == b"pre-existing"


class TestHasModelFilesPriority:
    """Document the precedence: safetensors wins over Orbax when both present."""

    def test_prefers_safetensors_over_orbax(self, tmp_path: Path) -> None:
        (tmp_path / "ocdbt.process_0").mkdir()
        (tmp_path / "manifest.ocdbt").write_bytes(b"")
        (tmp_path / "model.safetensors").write_bytes(b"")

        assert HubManager._has_safetensors(tmp_path) is True
        assert HubManager._has_orbax(tmp_path) is True
        assert HubManager._has_model_files(tmp_path) is True

    def test_orbax_only(self, tmp_path: Path) -> None:
        (tmp_path / "ocdbt.process_0").mkdir()
        (tmp_path / "manifest.ocdbt").write_bytes(b"")

        assert HubManager._has_safetensors(tmp_path) is False
        assert HubManager._has_orbax(tmp_path) is True
        assert HubManager._has_model_files(tmp_path) is True
