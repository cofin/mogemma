"""Tests for vision config.json parsing and wiring (Ch5)."""

import json
from pathlib import Path

import pytest

from mogemma.config import GenerationConfig
from mogemma.model import _parse_gemma4_architecture


@pytest.fixture
def gemma4_vision_config(tmp_path: Path) -> Path:
    """Create a minimal Gemma 4 config.json with vision_config section."""
    config = {
        "model_type": "gemma4",
        "num_hidden_layers": 4,
        "hidden_size": 3584,
        "sliding_window_size": 1024,
        "partial_rotary_factor": 0.5,
        "attention_k_eq_v": True,
        "layer_types": ["sliding", "sliding", "full", "sliding"],
        "image_token_index": 255999,
        "vision_config": {
            "num_hidden_layers": 27,
            "hidden_size": 1152,
            "num_attention_heads": 16,
            "intermediate_size": 4304,
            "patch_size": 16,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return tmp_path


class TestVisionConfigParsing:
    """Tests for vision config fields in _parse_gemma4_architecture."""

    def test_parses_num_vision_layers(self, gemma4_vision_config: Path) -> None:
        overrides, _ = _parse_gemma4_architecture(gemma4_vision_config)
        assert overrides["num_vision_layers"] == 27

    def test_parses_vision_hidden_size(self, gemma4_vision_config: Path) -> None:
        overrides, _ = _parse_gemma4_architecture(gemma4_vision_config)
        assert overrides["vision_hidden_size"] == 1152

    def test_parses_vision_num_heads(self, gemma4_vision_config: Path) -> None:
        overrides, _ = _parse_gemma4_architecture(gemma4_vision_config)
        assert overrides["vision_num_heads"] == 16

    def test_parses_vision_intermediate_size(self, gemma4_vision_config: Path) -> None:
        overrides, _ = _parse_gemma4_architecture(gemma4_vision_config)
        assert overrides["vision_intermediate_size"] == 4304

    def test_parses_image_token_id(self, gemma4_vision_config: Path) -> None:
        overrides, _ = _parse_gemma4_architecture(gemma4_vision_config)
        assert overrides["image_token_id"] == 255999

    def test_no_vision_section_no_vision_overrides(self, tmp_path: Path) -> None:
        config = {"model_type": "gemma4", "hidden_size": 3584}
        (tmp_path / "config.json").write_text(json.dumps(config))
        overrides, _ = _parse_gemma4_architecture(tmp_path)
        assert "num_vision_layers" not in overrides

    def test_still_parses_text_config_with_vision(self, gemma4_vision_config: Path) -> None:
        overrides, layer_types = _parse_gemma4_architecture(gemma4_vision_config)
        assert overrides["window_size"] == 1024
        assert overrides["k_eq_v"] == 1
        assert layer_types == [0, 0, 1, 0]


class TestGenerationConfigMaxImageTokens:
    """Test max_image_tokens field on GenerationConfig."""

    def test_default_max_image_tokens(self) -> None:
        config = GenerationConfig(model_path="google/gemma-4-31B-it")
        assert config.max_image_tokens == 560

    def test_custom_max_image_tokens(self) -> None:
        config = GenerationConfig(model_path="google/gemma-4-31B-it", max_image_tokens=1120)
        assert config.max_image_tokens == 1120
