"""Tests for the Orbax→safetensors converter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mogemma.convert import _layer_count, _variant_from_keys

if TYPE_CHECKING:
    pass


class TestVariantDetection:
    """`_variant_from_keys` classifies checkpoints by their Orbax key inventory."""

    def test_base_variant_has_no_ple_or_moe(self) -> None:
        keys = [
            "embedder.input_embedding",
            "final_norm.scale",
            "layer_0.attn.q_einsum.w",
            "layer_0.mlp.gating_einsum.w",
        ]
        assert _variant_from_keys(keys) == "base"

    def test_ple_variant_has_per_layer_embedder(self) -> None:
        keys = [
            "embedder.input_embedding",
            "embedder.per_layer_embeddings",
            "embedder.per_layer_model_projection.w",
            "layer_0.attn.q_einsum.w",
        ]
        assert _variant_from_keys(keys) == "ple"

    def test_moe_variant_has_router_logits(self) -> None:
        keys = [
            "embedder.input_embedding",
            "layer_0.attn.q_einsum.w",
            "layer_0.mlp.router_logits.w",
        ]
        assert _variant_from_keys(keys) == "moe"

    def test_moe_takes_priority_over_ple(self) -> None:
        """A hypothetical checkpoint with both router_logits and PLE keys is classified as MoE."""
        keys = [
            "embedder.per_layer_embeddings",
            "layer_0.mlp.router_logits.w",
        ]
        assert _variant_from_keys(keys) == "moe"


class TestLayerCount:
    """`_layer_count` extracts the number of transformer layers from key names."""

    def test_dense_layers(self) -> None:
        keys = [f"layer_{i}.attn.q_einsum.w" for i in range(5)]
        assert _layer_count(keys) == 5

    def test_nonzero_start_still_counted_as_max_plus_one(self) -> None:
        keys = ["layer_7.attn.q_einsum.w", "layer_0.attn.q_einsum.w"]
        assert _layer_count(keys) == 8

    def test_ignores_non_layer_keys(self) -> None:
        keys = [
            "embedder.input_embedding",
            "final_norm.scale",
            "layer_0.attn.q_einsum.w",
            "layer_1.attn.q_einsum.w",
        ]
        assert _layer_count(keys) == 2

    def test_raises_on_empty_input(self) -> None:
        with pytest.raises(ValueError, match="no layer_"):
            _layer_count(["embedder.input_embedding"])
