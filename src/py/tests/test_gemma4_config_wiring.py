"""Tests for Gemma 4 config.json parsing and architecture override wiring (Ch4)."""

import json
from pathlib import Path

import pytest

from mogemma.model import _parse_gemma4_architecture, compute_kv_cache_memory


@pytest.fixture
def gemma4_31b_config(tmp_path: Path) -> Path:
    """Create a minimal Gemma 4 31B config.json."""
    config = {
        "model_type": "gemma4",
        "num_hidden_layers": 4,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "intermediate_size": 256,
        "vocab_size": 1000,
        "sliding_window_size": 1024,
        "partial_rotary_factor": 0.5,
        "attention_k_eq_v": True,
        "final_logit_softcapping": 30.0,
        "attn_logit_softcapping": 50.0,
        "layer_types": ["sliding", "sliding", "full", "sliding"],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return tmp_path


@pytest.fixture
def gemma4_e2b_config(tmp_path: Path) -> Path:
    """Create a minimal Gemma 4 E2B config.json."""
    config = {
        "model_type": "gemma4",
        "num_hidden_layers": 3,
        "hidden_size": 64,
        "hidden_size_per_layer_input": 256,
        "use_double_wide_mlp": True,
        "sliding_window_size": 512,
        "partial_rotary_factor": 0.5,
        "attention_k_eq_v": False,
        "layer_types": ["sliding", "full", "sliding"],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return tmp_path


class TestParseGemma4Architecture:
    """Tests for _parse_gemma4_architecture config parsing."""

    def test_parses_window_size(self, gemma4_31b_config: Path) -> None:
        overrides, _ = _parse_gemma4_architecture(gemma4_31b_config)
        assert overrides["window_size"] == 1024

    def test_parses_partial_rotary_factor(self, gemma4_31b_config: Path) -> None:
        overrides, _ = _parse_gemma4_architecture(gemma4_31b_config)
        assert overrides["partial_rotary_factor"] == 0.5

    def test_parses_k_eq_v_true(self, gemma4_31b_config: Path) -> None:
        overrides, _ = _parse_gemma4_architecture(gemma4_31b_config)
        assert overrides["k_eq_v"] == 1

    def test_parses_k_eq_v_false(self, gemma4_e2b_config: Path) -> None:
        overrides, _ = _parse_gemma4_architecture(gemma4_e2b_config)
        assert overrides["k_eq_v"] == 0

    def test_parses_final_logit_softcapping(self, gemma4_31b_config: Path) -> None:
        overrides, _ = _parse_gemma4_architecture(gemma4_31b_config)
        assert overrides["final_logit_softcapping"] == 30.0

    def test_parses_attn_logit_softcapping(self, gemma4_31b_config: Path) -> None:
        overrides, _ = _parse_gemma4_architecture(gemma4_31b_config)
        assert overrides["attn_logit_softcapping"] == 50.0

    def test_parses_layer_types(self, gemma4_31b_config: Path) -> None:
        _, layer_types = _parse_gemma4_architecture(gemma4_31b_config)
        assert layer_types == [0, 0, 1, 0]  # sliding, sliding, full, sliding

    def test_e2b_smaller_window(self, gemma4_e2b_config: Path) -> None:
        overrides, _ = _parse_gemma4_architecture(gemma4_e2b_config)
        assert overrides["window_size"] == 512

    def test_e2b_layer_types(self, gemma4_e2b_config: Path) -> None:
        _, layer_types = _parse_gemma4_architecture(gemma4_e2b_config)
        assert layer_types == [0, 1, 0]

    def test_defaults_when_no_config(self, tmp_path: Path) -> None:
        overrides, layer_types = _parse_gemma4_architecture(tmp_path)
        assert overrides == {}
        assert layer_types == []

    def test_defaults_for_missing_fields(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"model_type": "gemma4"}))
        overrides, layer_types = _parse_gemma4_architecture(tmp_path)
        assert overrides["window_size"] == 1024
        assert overrides["partial_rotary_factor"] == 0.5
        assert overrides["k_eq_v"] == 0
        assert layer_types == []

    def test_missing_softcap_fields_are_not_defaulted(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"model_type": "gemma4"}))
        overrides, _ = _parse_gemma4_architecture(tmp_path)
        assert "final_logit_softcapping" not in overrides
        assert "attn_logit_softcapping" not in overrides

    def test_sliding_window_alias(self, tmp_path: Path) -> None:
        """Some configs use 'sliding_window' instead of 'sliding_window_size'."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"sliding_window": 512}))
        overrides, _ = _parse_gemma4_architecture(tmp_path)
        assert overrides["window_size"] == 512

    def test_empty_layer_types_returns_empty(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"model_type": "gemma4"}))
        _, layer_types = _parse_gemma4_architecture(tmp_path)
        assert layer_types == []


class TestComputeKvCacheMemoryStillWorks:
    """Verify compute_kv_cache_memory utility from Ch3 is not broken."""

    def test_basic_computation(self) -> None:
        bytes_total = compute_kv_cache_memory(
            num_layers=4,
            layer_types=["sliding", "sliding", "full", "sliding"],
            window_size=1024,
            max_context_len=8192,
            num_kv_heads=4,
            head_dim=256,
        )
        # 3 sliding layers: 3 * 1024 * 4 * 256 = 3,145,728 elements
        # 1 full layer: 1 * 8192 * 4 * 256 = 8,388,608 elements
        # total = 11,534,336 elements * 4 bytes * 2 (K+V) = 92,274,688 bytes
        assert bytes_total == 92_274_688

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="layer_types length"):
            compute_kv_cache_memory(
                num_layers=4,
                layer_types=["sliding", "sliding"],
                window_size=1024,
                max_context_len=8192,
                num_kv_heads=4,
                head_dim=256,
            )

    def test_context_less_than_window_raises(self) -> None:
        with pytest.raises(ValueError, match="max_context_len"):
            compute_kv_cache_memory(
                num_layers=2,
                layer_types=["sliding", "full"],
                window_size=2048,
                max_context_len=1024,
                num_kv_heads=4,
                head_dim=256,
            )
