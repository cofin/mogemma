"""Tests for Gemma 4 variant detection from config.json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mogemma.model import Gemma4Variant, _detect_gemma4_variant


@pytest.fixture()
def tmp_model_dir(tmp_path: Path) -> Path:
    """Return a temporary directory to use as a model directory."""
    return tmp_path


def _write_config(model_dir: Path, config: dict) -> Path:
    """Write a config.json to the model directory and return its path."""
    config_path = model_dir / "config.json"
    config_path.write_text(json.dumps(config))
    return config_path


class TestGemma4VariantEnum:
    def test_dense_31b_value(self) -> None:
        assert Gemma4Variant.DENSE_31B == "gemma4_dense_31b"

    def test_dense_e2b_value(self) -> None:
        assert Gemma4Variant.DENSE_E2B == "gemma4_dense_e2b"

    def test_dense_e4b_value(self) -> None:
        assert Gemma4Variant.DENSE_E4B == "gemma4_dense_e4b"

    def test_moe_26b_value(self) -> None:
        assert Gemma4Variant.MOE_26B_A4B == "gemma4_moe_26b"

    def test_is_string_enum(self) -> None:
        assert isinstance(Gemma4Variant.DENSE_31B, str)


class TestDetectGemma4Variant:
    def test_dense_31b_basic(self, tmp_model_dir: Path) -> None:
        """A config with no special fields should detect as DENSE_31B."""
        _write_config(tmp_model_dir, {"hidden_size": 4096})
        result = _detect_gemma4_variant(tmp_model_dir)
        assert result is Gemma4Variant.DENSE_31B

    def test_moe_26b(self, tmp_model_dir: Path) -> None:
        """Config with num_experts > 0 should detect as MOE_26B_A4B."""
        _write_config(tmp_model_dir, {"num_experts": 64})
        result = _detect_gemma4_variant(tmp_model_dir)
        assert result is Gemma4Variant.MOE_26B_A4B

    def test_moe_zero_experts_is_dense(self, tmp_model_dir: Path) -> None:
        """num_experts=0 should NOT be MOE."""
        _write_config(tmp_model_dir, {"num_experts": 0})
        result = _detect_gemma4_variant(tmp_model_dir)
        assert result is Gemma4Variant.DENSE_31B

    def test_dense_e2b(self, tmp_model_dir: Path) -> None:
        """PLE with double wide MLP should be DENSE_E2B."""
        _write_config(tmp_model_dir, {
            "hidden_size_per_layer_input": 2048,
            "use_double_wide_mlp": True,
        })
        result = _detect_gemma4_variant(tmp_model_dir)
        assert result is Gemma4Variant.DENSE_E2B

    def test_dense_e4b(self, tmp_model_dir: Path) -> None:
        """PLE without double wide MLP should be DENSE_E4B."""
        _write_config(tmp_model_dir, {
            "hidden_size_per_layer_input": 2048,
            "use_double_wide_mlp": False,
        })
        result = _detect_gemma4_variant(tmp_model_dir)
        assert result is Gemma4Variant.DENSE_E4B

    def test_dense_e4b_no_mlp_key(self, tmp_model_dir: Path) -> None:
        """PLE with missing use_double_wide_mlp defaults to E4B."""
        _write_config(tmp_model_dir, {"hidden_size_per_layer_input": 2048})
        result = _detect_gemma4_variant(tmp_model_dir)
        assert result is Gemma4Variant.DENSE_E4B

    def test_missing_config_json(self, tmp_model_dir: Path) -> None:
        """Should raise FileNotFoundError when config.json is missing."""
        with pytest.raises(FileNotFoundError):
            _detect_gemma4_variant(tmp_model_dir)
