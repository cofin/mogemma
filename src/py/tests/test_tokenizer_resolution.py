"""Tests for tokenizer resolution priority and config.json validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from mogemma.hub import HubManager


class TestTokenizerResolution:
    """Tests for tokenizer file resolution with priority ordering."""

    def test_hf_local_tokenizer_preferred(self, tmp_path: Path) -> None:
        """tokenizer.model in model dir should be found first."""
        (tmp_path / "tokenizer.model").write_bytes(b"fake-tokenizer")
        hub = HubManager(cache_path=tmp_path)
        path = hub.resolve_tokenizer(tmp_path)
        assert path == tmp_path / "tokenizer.model"

    def test_explicit_path_override(self, tmp_path: Path) -> None:
        """Explicit tokenizer_path should take highest priority."""
        explicit = tmp_path / "custom" / "my_tokenizer.model"
        explicit.parent.mkdir(parents=True)
        explicit.write_bytes(b"custom-tokenizer")
        hub = HubManager(cache_path=tmp_path)
        path = hub.resolve_tokenizer(tmp_path, tokenizer_path=explicit)
        assert path == explicit

    def test_raises_when_no_tokenizer_found(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError when no tokenizer exists."""
        hub = HubManager(cache_path=tmp_path)
        with pytest.raises(FileNotFoundError, match="tokenizer"):
            hub.resolve_tokenizer(tmp_path)

    def test_explicit_path_raises_if_missing(self, tmp_path: Path) -> None:
        """Explicit path that doesn't exist should raise."""
        hub = HubManager(cache_path=tmp_path)
        with pytest.raises(FileNotFoundError, match="tokenizer"):
            hub.resolve_tokenizer(tmp_path, tokenizer_path=tmp_path / "nonexistent.model")


class TestConfigJsonValidation:
    """Tests for config.json field validation."""

    def test_valid_gemma4_config(self) -> None:
        config = {
            "model_type": "gemma4",
            "num_hidden_layers": 60,
            "hidden_size": 4096,
        }
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        hub.validate_config_json(config)  # Should not raise

    def test_valid_gemma4_text_config(self) -> None:
        config = {
            "model_type": "gemma4_text",
            "num_hidden_layers": 60,
            "hidden_size": 4096,
        }
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        hub.validate_config_json(config)  # Should not raise

    def test_raises_on_missing_model_type(self) -> None:
        config = {"num_hidden_layers": 60}
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        with pytest.raises(ValueError, match="model_type"):
            hub.validate_config_json(config)

    def test_raises_on_non_gemma4_model_type(self) -> None:
        config = {"model_type": "llama", "num_hidden_layers": 32}
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        with pytest.raises(ValueError, match="Gemma 4"):
            hub.validate_config_json(config)

    def test_raises_on_missing_num_hidden_layers(self) -> None:
        config = {"model_type": "gemma4"}
        hub = HubManager(cache_path=Path("/tmp/test-cache"))
        with pytest.raises(ValueError, match="num_hidden_layers"):
            hub.validate_config_json(config)
