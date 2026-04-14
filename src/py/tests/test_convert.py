"""Tests for the Orbax→safetensors converter."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Callable
from unittest.mock import patch

import numpy as np
import pytest

from mogemma.convert import _iter_base_transformer, _layer_count, _variant_from_keys, _write_sharded
from mogemma.orbax_loader import OrbaxLoader

if TYPE_CHECKING:
    from pathlib import Path


# E2B-it shapes, shrunk for fast synthetic tests but preserving axis roles.
_V, _H, _N_LAYERS, _N_HEADS, _N_KV_HEADS, _HEAD_DIM, _INTER = 128, 96, 2, 4, 1, 24, 256


def _make_base_tensor_map(num_layers: int = _N_LAYERS) -> dict[str, np.ndarray]:
    """Build a synthetic Orbax tensor dict with E2B-it axis roles."""
    rng = np.random.default_rng(seed=0)
    tensors: dict[str, np.ndarray] = {
        "embedder.input_embedding": rng.standard_normal((_V, _H)).astype(np.float32),
        "final_norm.scale": rng.standard_normal((_H,)).astype(np.float32),
    }
    for n in range(num_layers):
        tensors[f"layer_{n}.attn.q_einsum.w"] = rng.standard_normal((_N_HEADS, _H, _HEAD_DIM)).astype(np.float32)
        tensors[f"layer_{n}.attn.kv_einsum.w"] = rng.standard_normal((2, _N_KV_HEADS, _H, _HEAD_DIM)).astype(np.float32)
        tensors[f"layer_{n}.attn.attn_vec_einsum.w"] = rng.standard_normal((_N_HEADS, _HEAD_DIM, _H)).astype(np.float32)
        tensors[f"layer_{n}.attn.query_norm.scale"] = rng.standard_normal((_HEAD_DIM,)).astype(np.float32)
        tensors[f"layer_{n}.attn.key_norm.scale"] = rng.standard_normal((_HEAD_DIM,)).astype(np.float32)
        tensors[f"layer_{n}.mlp.gating_einsum.w"] = rng.standard_normal((2, _INTER, _H)).astype(np.float32)
        tensors[f"layer_{n}.mlp.linear.w"] = rng.standard_normal((_INTER, _H)).astype(np.float32)
        tensors[f"layer_{n}.pre_attention_norm.scale"] = rng.standard_normal((_H,)).astype(np.float32)
        tensors[f"layer_{n}.post_attention_norm.scale"] = rng.standard_normal((_H,)).astype(np.float32)
        tensors[f"layer_{n}.pre_ffw_norm.scale"] = rng.standard_normal((_H,)).astype(np.float32)
        tensors[f"layer_{n}.post_ffw_norm.scale"] = rng.standard_normal((_H,)).astype(np.float32)
    return tensors


def _run_iter_with_fakes(
    tensors: dict[str, np.ndarray], iterator_fn: Callable[..., "Iterator[tuple[str, np.ndarray]]"], *args: object
) -> list[tuple[str, np.ndarray]]:
    """Patch OrbaxLoader.open_tensor to pull from *tensors*, run *iterator_fn*, collect results."""

    def fake_open_tensor(_path: object, name: str) -> np.ndarray:
        return tensors[name]

    with patch.object(OrbaxLoader, "open_tensor", staticmethod(fake_open_tensor)):
        return list(iterator_fn(*args))


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
        keys = ["embedder.input_embedding", "layer_0.attn.q_einsum.w", "layer_0.mlp.router_logits.w"]
        assert _variant_from_keys(keys) == "moe"

    def test_moe_takes_priority_over_ple(self) -> None:
        """A hypothetical checkpoint with both router_logits and PLE keys is classified as MoE."""
        keys = ["embedder.per_layer_embeddings", "layer_0.mlp.router_logits.w"]
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
        keys = ["embedder.input_embedding", "final_norm.scale", "layer_0.attn.q_einsum.w", "layer_1.attn.q_einsum.w"]
        assert _layer_count(keys) == 2

    def test_raises_on_empty_input(self) -> None:
        with pytest.raises(ValueError, match="no layer_"):
            _layer_count(["embedder.input_embedding"])


class TestBaseTransformerIterator:
    """`_iter_base_transformer` yields the Mojo-contract tensor names + transformed arrays."""

    def test_yields_expected_name_set(self, tmp_path: Path) -> None:
        tensors = _make_base_tensor_map()
        keys = list(tensors.keys())
        yielded = _run_iter_with_fakes(tensors, _iter_base_transformer, tmp_path, keys, _N_LAYERS)
        names = {name for name, _ in yielded}

        expected = {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"}
        for n in range(_N_LAYERS):
            pfx = f"model.layers.{n}"
            expected |= {
                f"{pfx}.input_layernorm.weight",
                f"{pfx}.post_attention_layernorm.weight",
                f"{pfx}.pre_feedforward_layernorm.weight",
                f"{pfx}.post_feedforward_layernorm.weight",
                f"{pfx}.self_attn.q_proj.weight",
                f"{pfx}.self_attn.k_proj.weight",
                f"{pfx}.self_attn.v_proj.weight",
                f"{pfx}.self_attn.o_proj.weight",
                f"{pfx}.self_attn.q_norm.weight",
                f"{pfx}.self_attn.k_norm.weight",
                f"{pfx}.mlp.gate_proj.weight",
                f"{pfx}.mlp.up_proj.weight",
                f"{pfx}.mlp.down_proj.weight",
            }
        assert names == expected

    def test_embed_tokens_and_lm_head_are_tied(self, tmp_path: Path) -> None:
        tensors = _make_base_tensor_map()
        yielded = dict(_run_iter_with_fakes(tensors, _iter_base_transformer, tmp_path, list(tensors), _N_LAYERS))
        np.testing.assert_array_equal(yielded["model.embed_tokens.weight"], yielded["lm_head.weight"])
        np.testing.assert_array_equal(yielded["model.embed_tokens.weight"], tensors["embedder.input_embedding"])

    def test_q_proj_transform(self, tmp_path: Path) -> None:
        tensors = _make_base_tensor_map()
        yielded = dict(_run_iter_with_fakes(tensors, _iter_base_transformer, tmp_path, list(tensors), _N_LAYERS))
        src = tensors["layer_0.attn.q_einsum.w"]  # (n_heads, H, head_dim)
        expected = src.transpose(0, 2, 1).reshape(-1, _H)  # (n_heads*head_dim, H)
        np.testing.assert_array_equal(yielded["model.layers.0.self_attn.q_proj.weight"], expected)
        assert yielded["model.layers.0.self_attn.q_proj.weight"].shape == (_N_HEADS * _HEAD_DIM, _H)

    def test_kv_proj_split_and_transform(self, tmp_path: Path) -> None:
        tensors = _make_base_tensor_map()
        yielded = dict(_run_iter_with_fakes(tensors, _iter_base_transformer, tmp_path, list(tensors), _N_LAYERS))
        src = tensors["layer_0.attn.kv_einsum.w"]  # (2, n_kv, H, head_dim)
        expected_k = src[0].transpose(0, 2, 1).reshape(-1, _H)
        expected_v = src[1].transpose(0, 2, 1).reshape(-1, _H)
        np.testing.assert_array_equal(yielded["model.layers.0.self_attn.k_proj.weight"], expected_k)
        np.testing.assert_array_equal(yielded["model.layers.0.self_attn.v_proj.weight"], expected_v)

    def test_o_proj_transform(self, tmp_path: Path) -> None:
        tensors = _make_base_tensor_map()
        yielded = dict(_run_iter_with_fakes(tensors, _iter_base_transformer, tmp_path, list(tensors), _N_LAYERS))
        src = tensors["layer_0.attn.attn_vec_einsum.w"]  # (n_heads, head_dim, H)
        expected = src.reshape(-1, _H).T  # (H, n_heads*head_dim)
        np.testing.assert_array_equal(yielded["model.layers.0.self_attn.o_proj.weight"], expected)
        assert yielded["model.layers.0.self_attn.o_proj.weight"].shape == (_H, _N_HEADS * _HEAD_DIM)

    def test_mlp_gate_up_split(self, tmp_path: Path) -> None:
        tensors = _make_base_tensor_map()
        yielded = dict(_run_iter_with_fakes(tensors, _iter_base_transformer, tmp_path, list(tensors), _N_LAYERS))
        src = tensors["layer_0.mlp.gating_einsum.w"]  # (2, intermediate, H)
        np.testing.assert_array_equal(yielded["model.layers.0.mlp.gate_proj.weight"], src[0])
        np.testing.assert_array_equal(yielded["model.layers.0.mlp.up_proj.weight"], src[1])

    def test_mlp_down_proj_transposes(self, tmp_path: Path) -> None:
        tensors = _make_base_tensor_map()
        yielded = dict(_run_iter_with_fakes(tensors, _iter_base_transformer, tmp_path, list(tensors), _N_LAYERS))
        src = tensors["layer_0.mlp.linear.w"]  # (intermediate, H) in Orbax
        np.testing.assert_array_equal(yielded["model.layers.0.mlp.down_proj.weight"], src.T)
        assert yielded["model.layers.0.mlp.down_proj.weight"].shape == (_H, _INTER)

    def test_norms_pass_through_unchanged(self, tmp_path: Path) -> None:
        tensors = _make_base_tensor_map()
        yielded = dict(_run_iter_with_fakes(tensors, _iter_base_transformer, tmp_path, list(tensors), _N_LAYERS))
        np.testing.assert_array_equal(yielded["model.norm.weight"], tensors["final_norm.scale"])
        np.testing.assert_array_equal(
            yielded["model.layers.0.input_layernorm.weight"], tensors["layer_0.pre_attention_norm.scale"]
        )
        np.testing.assert_array_equal(
            yielded["model.layers.0.post_attention_layernorm.weight"], tensors["layer_0.post_attention_norm.scale"]
        )
        np.testing.assert_array_equal(
            yielded["model.layers.0.pre_feedforward_layernorm.weight"], tensors["layer_0.pre_ffw_norm.scale"]
        )
        np.testing.assert_array_equal(
            yielded["model.layers.0.post_feedforward_layernorm.weight"], tensors["layer_0.post_ffw_norm.scale"]
        )
        np.testing.assert_array_equal(
            yielded["model.layers.0.self_attn.q_norm.weight"], tensors["layer_0.attn.query_norm.scale"]
        )
        np.testing.assert_array_equal(
            yielded["model.layers.0.self_attn.k_norm.weight"], tensors["layer_0.attn.key_norm.scale"]
        )


class TestWriteSharded:
    """`_write_sharded` writes safetensors shards + index.json."""

    def test_single_shard_under_threshold(self, tmp_path: Path) -> None:
        tensors = [
            ("model.embed_tokens.weight", np.ones((4, 4), dtype=np.float32)),
            ("model.norm.weight", np.ones((4,), dtype=np.float32)),
        ]
        written = _write_sharded(tmp_path, iter(tensors), shard_size_bytes=1024 * 1024)

        assert written == [tmp_path / "model.safetensors"]
        assert (tmp_path / "model.safetensors").exists()
        # No index for single-shard output.
        assert not (tmp_path / "model.safetensors.index.json").exists()

        from safetensors import safe_open  # type: ignore[import-untyped]

        with safe_open(str(tmp_path / "model.safetensors"), framework="numpy") as f:  # type: ignore[no-untyped-call]
            assert set(f.keys()) == {"model.embed_tokens.weight", "model.norm.weight"}

    def test_multi_shard_produces_index(self, tmp_path: Path) -> None:
        # Each tensor is 16KB; threshold at 20KB forces 3 shards (one tensor per shard).
        tensors = [
            ("a", np.ones((64, 64), dtype=np.float32)),
            ("b", np.ones((64, 64), dtype=np.float32)),
            ("c", np.ones((64, 64), dtype=np.float32)),
        ]
        written = _write_sharded(tmp_path, iter(tensors), shard_size_bytes=20 * 1024)

        assert len(written) == 3
        shard_names = [p.name for p in written]
        assert shard_names == [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ]
        index_path = tmp_path / "model.safetensors.index.json"
        assert index_path.exists()

        import json

        index = json.loads(index_path.read_text())
        assert set(index["weight_map"].keys()) == {"a", "b", "c"}
        # Each tensor lives in a distinct shard.
        assert len(set(index["weight_map"].values())) == 3
        assert index["metadata"]["total_size"] > 0

    def test_multi_shard_index_points_to_correct_files(self, tmp_path: Path) -> None:
        tensors = [("first", np.ones((64, 64), dtype=np.float32)), ("second", np.ones((64, 64), dtype=np.float32))]
        _write_sharded(tmp_path, iter(tensors), shard_size_bytes=20 * 1024)

        import json

        from safetensors import safe_open  # type: ignore[import-untyped]

        index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
        for name in ("first", "second"):
            shard = index["weight_map"][name]
            with safe_open(str(tmp_path / shard), framework="numpy") as f:  # type: ignore[no-untyped-call]
                assert name in list(f.keys())

    def test_tensor_larger_than_threshold_gets_own_shard(self, tmp_path: Path) -> None:
        """A single tensor bigger than shard_size_bytes still writes, in its own shard."""
        tensors = [
            ("small", np.ones((4, 4), dtype=np.float32)),
            ("big", np.ones((128, 128), dtype=np.float32)),  # 64KB
        ]
        written = _write_sharded(tmp_path, iter(tensors), shard_size_bytes=10 * 1024)
        assert len(written) == 2
