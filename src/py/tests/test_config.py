"""Tests for Gemma 4 configuration defaults and validation."""

from __future__ import annotations

import pytest

from mogemma.config import EmbeddingConfig, GenerationConfig


class TestGenerationConfigDefaults:
    def test_default_model_path(self) -> None:
        config = GenerationConfig()
        assert str(config.model_path) == "google/gemma-4-26B-A4B-it"

    def test_default_top_k(self) -> None:
        config = GenerationConfig()
        assert config.top_k == 64

    def test_default_top_p(self) -> None:
        config = GenerationConfig()
        assert config.top_p == 0.95

    def test_default_temperature(self) -> None:
        config = GenerationConfig()
        assert config.temperature == 1.0

    def test_default_max_tokens(self) -> None:
        config = GenerationConfig()
        assert config.max_tokens == 128


class TestEmbeddingConfigDefaults:
    def test_default_model_path(self) -> None:
        config = EmbeddingConfig()
        assert str(config.model_path) == "google/gemma-4-26B-A4B-it"


class TestGenerationConfigValidation:
    def test_negative_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            GenerationConfig(temperature=-1.0)

    def test_negative_top_k_raises(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            GenerationConfig(top_k=-1)

    def test_top_p_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="top_p"):
            GenerationConfig(top_p=1.5)

    def test_empty_model_path_raises(self) -> None:
        with pytest.raises(ValueError):
            GenerationConfig(model_path="")

    def test_zero_max_tokens_raises(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            GenerationConfig(max_tokens=0)
