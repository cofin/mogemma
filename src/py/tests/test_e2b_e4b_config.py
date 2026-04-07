"""Tests for E2B/E4B and MoE config parsing (Ch6/Ch7)."""

import json
from pathlib import Path

import pytest

from mogemma.model import Gemma4Variant, _detect_gemma4_variant, _parse_gemma4_architecture


@pytest.fixture
def e2b_config(tmp_path: Path) -> Path:
    config = {
        "model_type": "gemma4",
        "num_hidden_layers": 35,
        "hidden_size": 1536,
        "hidden_size_per_layer_input": 256,
        "vocab_size_per_layer_input": 262144,
        "use_double_wide_mlp": True,
        "sliding_window_size": 512,
        "partial_rotary_factor": 0.5,
        "attention_k_eq_v": False,
        "layer_types": ["sliding"] * 35,
        "kv_sharing_layer_map": [-1] * 15 + [i % 15 for i in range(20)],
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


@pytest.fixture
def e4b_config(tmp_path: Path) -> Path:
    config = {
        "model_type": "gemma4",
        "num_hidden_layers": 42,
        "hidden_size": 2560,
        "hidden_size_per_layer_input": 256,
        "vocab_size_per_layer_input": 262144,
        "use_double_wide_mlp": False,
        "sliding_window_size": 512,
        "attention_k_eq_v": False,
        "layer_types": ["sliding"] * 42,
        "kv_sharing_layer_map": [-1] * 24 + list(range(18)),
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


@pytest.fixture
def moe_26b_config(tmp_path: Path) -> Path:
    config = {
        "model_type": "gemma4",
        "num_hidden_layers": 30,
        "hidden_size": 3584,
        "num_local_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 704,
        "sliding_window_size": 1024,
        "partial_rotary_factor": 0.5,
        "attention_k_eq_v": True,
        "layer_types": ["sliding"] * 30,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


class TestE2BConfig:
    def test_parses_ple_dim(self, e2b_config: Path) -> None:
        overrides, _ = _parse_gemma4_architecture(e2b_config)
        assert overrides["hidden_size_per_layer_input"] == 256

    def test_parses_double_wide_mlp(self, e2b_config: Path) -> None:
        overrides, _ = _parse_gemma4_architecture(e2b_config)
        assert overrides["use_double_wide_mlp"] == 1

    def test_parses_kv_sharing(self, e2b_config: Path) -> None:
        overrides, _ = _parse_gemma4_architecture(e2b_config)
        assert overrides["kv_sharing_layer_count"] == 35

    def test_detects_e2b_variant(self, e2b_config: Path) -> None:
        variant = _detect_gemma4_variant(e2b_config)
        assert variant == Gemma4Variant.DENSE_E2B


class TestE4BConfig:
    def test_parses_ple_dim(self, e4b_config: Path) -> None:
        overrides, _ = _parse_gemma4_architecture(e4b_config)
        assert overrides["hidden_size_per_layer_input"] == 256

    def test_no_double_wide_mlp(self, e4b_config: Path) -> None:
        overrides, _ = _parse_gemma4_architecture(e4b_config)
        assert "use_double_wide_mlp" not in overrides

    def test_detects_e4b_variant(self, e4b_config: Path) -> None:
        variant = _detect_gemma4_variant(e4b_config)
        assert variant == Gemma4Variant.DENSE_E4B


class TestMoE26BConfig:
    def test_parses_num_experts(self, moe_26b_config: Path) -> None:
        overrides, _ = _parse_gemma4_architecture(moe_26b_config)
        assert overrides["num_experts"] == 128

    def test_parses_moe_top_k(self, moe_26b_config: Path) -> None:
        overrides, _ = _parse_gemma4_architecture(moe_26b_config)
        assert overrides["moe_top_k"] == 8

    def test_parses_moe_intermediate_size(self, moe_26b_config: Path) -> None:
        overrides, _ = _parse_gemma4_architecture(moe_26b_config)
        assert overrides["moe_intermediate_size"] == 704

    def test_detects_moe_variant(self, moe_26b_config: Path) -> None:
        variant = _detect_gemma4_variant(moe_26b_config)
        assert variant == Gemma4Variant.MOE_26B_A4B

    def test_k_eq_v_for_moe(self, moe_26b_config: Path) -> None:
        overrides, _ = _parse_gemma4_architecture(moe_26b_config)
        assert overrides["k_eq_v"] == 1
