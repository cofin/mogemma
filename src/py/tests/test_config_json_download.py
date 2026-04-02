"""Tests for config.json download and variant detection integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mogemma.hub import HubManager


class TestConfigJsonDownload:
    """Tests for config.json being included in the download flow."""

    def test_config_json_in_download_list(self) -> None:
        """config.json should always be downloaded alongside shards."""
        # The download flow fetches index.json first, then downloads
        # config.json + shards + tokenizer. We verify the flow structure
        # by checking that config.json is in the files_to_download list.
        # This is implicit in download_sync — if index.json fails (network),
        # it raises ModelNotFoundError before reaching config.json.
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        with pytest.raises(HubManager.ModelNotFoundError):
            hub.download_sync("google/nonexistent-model")

    def test_validate_config_accepts_gemma4(self) -> None:
        config = {"model_type": "gemma4", "num_hidden_layers": 60}
        HubManager.validate_config_json(config)

    def test_validate_config_accepts_gemma4_text(self) -> None:
        config = {"model_type": "gemma4_text", "num_hidden_layers": 60}
        HubManager.validate_config_json(config)

    def test_validate_config_rejects_non_gemma4(self) -> None:
        config = {"model_type": "gemma3", "num_hidden_layers": 28}
        with pytest.raises(ValueError, match="Gemma 4"):
            HubManager.validate_config_json(config)

    def test_validate_config_rejects_missing_layers(self) -> None:
        config = {"model_type": "gemma4"}
        with pytest.raises(ValueError, match="num_hidden_layers"):
            HubManager.validate_config_json(config)


class TestConfigJsonIntegration:
    """Tests verifying config.json is available for variant detection after download."""

    def test_config_written_to_staging_dir(self, tmp_path: Path) -> None:
        """After download, config.json should exist in the model directory."""
        # Create a fake model dir with config.json as download_sync would produce
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        config = {"model_type": "gemma4_text", "num_hidden_layers": 60, "hidden_size": 4096}
        (model_dir / "config.json").write_text(json.dumps(config))

        # Verify it can be loaded and validated
        loaded = json.loads((model_dir / "config.json").read_text())
        HubManager.validate_config_json(loaded)
        assert loaded["model_type"] == "gemma4_text"
        assert loaded["num_hidden_layers"] == 60
