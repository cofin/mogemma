"""Tests for the Orbax→safetensors converter."""

from __future__ import annotations

import gc
import json
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import numpy as np
import pytest

from mogemma.convert import (
    _iter_base_transformer,
    _iter_moe_transformer,
    _iter_ple,
    _iter_vision,
    _layer_count,
    _variant_from_keys,
    _write_sharded,
)
from mogemma.orbax_loader import OrbaxLoader

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


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
    tensors: dict[str, np.ndarray], iterator_fn: Callable[..., Iterator[tuple[str, np.ndarray]]], *args: object
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

    def test_unified_12b_orbax_inventory_requires_validation(self) -> None:
        with pytest.raises(
            ValueError, match="Gemma 4 12B unified Orbax conversion requires a validated tensor inventory"
        ):
            _variant_from_keys(["unified_encoder.some_new_tensor"])


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

    def test_flushed_shards_do_not_retain_array_dicts(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """After a shard flush, only tensor names should be retained for the final index."""
        from safetensors import numpy as safetensors_numpy  # type: ignore[import-untyped]

        first_array_ref: weakref.ReferenceType[np.ndarray] | None = None
        save_calls = 0

        def fake_save_file(current: dict[str, np.ndarray], path: str) -> None:
            nonlocal save_calls
            save_calls += 1
            if save_calls == 2:
                gc.collect()
                assert first_array_ref is not None
                assert first_array_ref() is None
            Path(path).write_bytes(b"stub")

        def tensor_iter() -> Iterator[tuple[str, np.ndarray]]:
            nonlocal first_array_ref
            first = np.ones((64, 64), dtype=np.float32)
            first_array_ref = weakref.ref(first)
            yield "first", first
            second = np.ones((64, 64), dtype=np.float32)
            yield "second", second

        monkeypatch.setattr(safetensors_numpy, "save_file", fake_save_file)

        _write_sharded(tmp_path, tensor_iter(), shard_size_bytes=20 * 1024)


# E2B-it PLE axis roles (shrunk): embeddings are (V, L, ple_dim).
_PLE_DIM = 8


def _make_ple_tensor_map(num_layers: int = _N_LAYERS) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed=1)
    tensors: dict[str, np.ndarray] = {
        # Global embedder table: layer axis is the MIDDLE axis, per E2B-it inventory.
        "embedder.per_layer_embeddings": rng.standard_normal((_V, num_layers, _PLE_DIM)).astype(np.float32)
    }
    for n in range(num_layers):
        tensors[f"layer_{n}.per_layer_projection.w"] = rng.standard_normal((_PLE_DIM, _H)).astype(np.float32)
        tensors[f"layer_{n}.post_per_layer_input_norm.scale"] = rng.standard_normal((_H,)).astype(np.float32)
    return tensors


class TestPLEIterator:
    """`_iter_ple` maps the 3 Mojo-consumed PLE tensors per layer."""

    def test_yields_expected_name_set(self, tmp_path: Path) -> None:
        tensors = _make_ple_tensor_map()
        yielded = _run_iter_with_fakes(tensors, _iter_ple, tmp_path, _N_LAYERS)
        names = {name for name, _ in yielded}

        expected = set()
        for n in range(_N_LAYERS):
            pfx = f"model.layers.{n}.per_layer_input"
            expected |= {
                f"{pfx}.per_layer_embedding.weight",
                f"{pfx}.per_layer_projection.weight",
                f"{pfx}.per_layer_norm.weight",
            }
        assert names == expected

    def test_embedding_split_along_middle_axis(self, tmp_path: Path) -> None:
        tensors = _make_ple_tensor_map()
        yielded = dict(_run_iter_with_fakes(tensors, _iter_ple, tmp_path, _N_LAYERS))

        src = tensors["embedder.per_layer_embeddings"]  # (V, L, ple_dim)
        for n in range(_N_LAYERS):
            emitted = yielded[f"model.layers.{n}.per_layer_input.per_layer_embedding.weight"]
            assert emitted.shape == (_V, _PLE_DIM)
            np.testing.assert_array_equal(emitted, src[:, n, :])

    def test_projection_passes_through(self, tmp_path: Path) -> None:
        tensors = _make_ple_tensor_map()
        yielded = dict(_run_iter_with_fakes(tensors, _iter_ple, tmp_path, _N_LAYERS))
        emitted = yielded["model.layers.0.per_layer_input.per_layer_projection.weight"]
        np.testing.assert_array_equal(emitted, tensors["layer_0.per_layer_projection.w"])
        assert emitted.shape == (_PLE_DIM, _H)

    def test_norm_passes_through(self, tmp_path: Path) -> None:
        tensors = _make_ple_tensor_map()
        yielded = dict(_run_iter_with_fakes(tensors, _iter_ple, tmp_path, _N_LAYERS))
        emitted = yielded["model.layers.0.per_layer_input.per_layer_norm.weight"]
        np.testing.assert_array_equal(emitted, tensors["layer_0.post_per_layer_input_norm.scale"])
        assert emitted.shape == (_H,)

    def test_skipped_orbax_tensors_not_opened(self, tmp_path: Path) -> None:
        """input_gate, skip_scale, model_projection, projection_norm are NOT consumed — verify."""
        tensors = _make_ple_tensor_map()
        # Add the 4 skip-list tensors; _iter_ple must NOT read them.
        tensors["embedder.per_layer_model_projection.w"] = np.ones((_H, _N_LAYERS, _PLE_DIM), dtype=np.float32)
        tensors["embedder.per_layer_projection_norm.scale"] = np.ones((_PLE_DIM,), dtype=np.float32)
        tensors["layer_0.per_layer_input_gate.w"] = np.ones((_H, _PLE_DIM), dtype=np.float32)
        tensors["layer_0.skip_scale"] = np.ones((1,), dtype=np.float32)

        opened: list[str] = []

        def tracking_open(_path: object, name: str) -> np.ndarray:
            opened.append(name)
            return tensors[name]

        with patch.object(OrbaxLoader, "open_tensor", staticmethod(tracking_open)):
            list(_iter_ple(tmp_path, _N_LAYERS))

        skip_list = {
            "embedder.per_layer_model_projection.w",
            "embedder.per_layer_projection_norm.scale",
            "layer_0.per_layer_input_gate.w",
            "layer_0.skip_scale",
        }
        assert skip_list.isdisjoint(set(opened))


# 26B-A4B-it MoE shapes, shrunk for unit tests. Real dims per .agents/specs/
# orbax-safetensors-conversion/26b-inventory.txt:
#   q_einsum.w (16, 2816, 256), kv_einsum.w (2, 8, 2816, 256),
#   mlp.gating_einsum.w (128, 2, 704, 2816) — packed experts
#   mlp.linear.w (128, 704, 2816) — experts down
#   mlp2.gating_einsum.w (2, 2112, 2816), mlp2.linear.w (2112, 2816) — dense
#   mlp.router_logits.w (2816, 128), router_scale (2816,), per_expert_scale (128,)
_M_H = 64
_M_HEAD_DIM = 16
_M_N_HEADS = 4
_M_N_KV = 2
_M_N_EXPERTS = 6
_M_I_MOE = 20
_M_I_DENSE = 32
_M_LAYERS = 2


def _make_moe_tensor_map(num_layers: int = _M_LAYERS) -> dict[str, np.ndarray]:
    """Synthetic MoE Orbax inventory mirroring 26B-A4B-it axis roles."""
    rng = np.random.default_rng(seed=3)
    tensors: dict[str, np.ndarray] = {
        "embedder.input_embedding": rng.standard_normal((128, _M_H)).astype(np.float32),
        "final_norm.scale": rng.standard_normal((_M_H,)).astype(np.float32),
    }
    for n in range(num_layers):
        # Attention (sliding layout — kv_einsum combined)
        tensors[f"layer_{n}.attn.q_einsum.w"] = rng.standard_normal((_M_N_HEADS, _M_H, _M_HEAD_DIM)).astype(np.float32)
        tensors[f"layer_{n}.attn.kv_einsum.w"] = rng.standard_normal((2, _M_N_KV, _M_H, _M_HEAD_DIM)).astype(np.float32)
        tensors[f"layer_{n}.attn.attn_vec_einsum.w"] = rng.standard_normal((_M_N_HEADS, _M_HEAD_DIM, _M_H)).astype(
            np.float32
        )
        tensors[f"layer_{n}.attn.query_norm.scale"] = rng.standard_normal((_M_HEAD_DIM,)).astype(np.float32)
        tensors[f"layer_{n}.attn.key_norm.scale"] = rng.standard_normal((_M_HEAD_DIM,)).astype(np.float32)

        # Dense MLP branch
        tensors[f"layer_{n}.mlp2.gating_einsum.w"] = rng.standard_normal((2, _M_I_DENSE, _M_H)).astype(np.float32)
        tensors[f"layer_{n}.mlp2.linear.w"] = rng.standard_normal((_M_I_DENSE, _M_H)).astype(np.float32)

        # MoE branch
        tensors[f"layer_{n}.mlp.router_logits.w"] = rng.standard_normal((_M_H, _M_N_EXPERTS)).astype(np.float32)
        tensors[f"layer_{n}.mlp.router_scale"] = rng.standard_normal((_M_H,)).astype(np.float32)
        tensors[f"layer_{n}.mlp.per_expert_scale"] = rng.standard_normal((_M_N_EXPERTS,)).astype(np.float32)
        tensors[f"layer_{n}.mlp.gating_einsum.w"] = rng.standard_normal((_M_N_EXPERTS, 2, _M_I_MOE, _M_H)).astype(
            np.float32
        )
        tensors[f"layer_{n}.mlp.linear.w"] = rng.standard_normal((_M_N_EXPERTS, _M_I_MOE, _M_H)).astype(np.float32)

        # Norms — the full 7 observed in 26B-A4B-it
        for norm in (
            "pre_attention_norm",
            "post_attention_norm",
            "pre_ffw_norm",
            "post_ffw1_norm",
            "pre_ffw2_norm",
            "post_ffw2_norm",
            "post_ffw_norm",
        ):
            tensors[f"layer_{n}.{norm}.scale"] = rng.standard_normal((_M_H,)).astype(np.float32)

        # skip_scale scalar
        tensors[f"layer_{n}.skip_scale"] = rng.standard_normal((1,)).astype(np.float32)
    return tensors


class TestMoEIterator:
    """`_iter_moe_transformer` emits the two-branch tensors consumed by the Mojo MoE forward pass."""

    def test_yields_expected_name_set(self, tmp_path: Path) -> None:
        tensors = _make_moe_tensor_map()
        keys = list(tensors)
        yielded = _run_iter_with_fakes(tensors, _iter_moe_transformer, tmp_path, keys, _M_LAYERS)
        names = {name for name, _ in yielded}

        expected = {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"}
        for n in range(_M_LAYERS):
            pfx = f"model.layers.{n}"
            expected |= {
                f"{pfx}.input_layernorm.weight",
                f"{pfx}.post_attention_layernorm.weight",
                f"{pfx}.self_attn.q_proj.weight",
                f"{pfx}.self_attn.k_proj.weight",
                f"{pfx}.self_attn.v_proj.weight",
                f"{pfx}.self_attn.o_proj.weight",
                f"{pfx}.self_attn.q_norm.weight",
                f"{pfx}.self_attn.k_norm.weight",
                # Dense branch
                f"{pfx}.mlp.gate_proj.weight",
                f"{pfx}.mlp.up_proj.weight",
                f"{pfx}.mlp.down_proj.weight",
                # Router
                f"{pfx}.moe_router.proj.weight",
                f"{pfx}.moe_router.scale",
                f"{pfx}.moe_router.per_expert_scale",
                # Experts (packed)
                f"{pfx}.moe_experts.gate_up_proj",
                f"{pfx}.moe_experts.down_proj",
                # MoE norms
                f"{pfx}.pre_feedforward_layernorm.weight",
                f"{pfx}.post_feedforward_layernorm_1.weight",
                f"{pfx}.pre_feedforward_layernorm_2.weight",
                f"{pfx}.post_feedforward_layernorm_2.weight",
                f"{pfx}.post_feedforward_layernorm.weight",
                f"{pfx}.moe_skip_scale.weight",
            }
        assert names == expected

    def test_dense_branch_splits_mlp2_gating(self, tmp_path: Path) -> None:
        tensors = _make_moe_tensor_map()
        yielded = dict(_run_iter_with_fakes(tensors, _iter_moe_transformer, tmp_path, list(tensors), _M_LAYERS))
        src = tensors["layer_0.mlp2.gating_einsum.w"]  # (2, I_dense, H)
        np.testing.assert_array_equal(yielded["model.layers.0.mlp.gate_proj.weight"], src[0])
        np.testing.assert_array_equal(yielded["model.layers.0.mlp.up_proj.weight"], src[1])

    def test_dense_down_transposes(self, tmp_path: Path) -> None:
        tensors = _make_moe_tensor_map()
        yielded = dict(_run_iter_with_fakes(tensors, _iter_moe_transformer, tmp_path, list(tensors), _M_LAYERS))
        src = tensors["layer_0.mlp2.linear.w"]  # (I_dense, H)
        np.testing.assert_array_equal(yielded["model.layers.0.mlp.down_proj.weight"], src.T)
        assert yielded["model.layers.0.mlp.down_proj.weight"].shape == (_M_H, _M_I_DENSE)

    def test_router_transpose(self, tmp_path: Path) -> None:
        tensors = _make_moe_tensor_map()
        yielded = dict(_run_iter_with_fakes(tensors, _iter_moe_transformer, tmp_path, list(tensors), _M_LAYERS))
        src = tensors["layer_0.mlp.router_logits.w"]  # (H, E)
        np.testing.assert_array_equal(yielded["model.layers.0.moe_router.proj.weight"], src.T)
        assert yielded["model.layers.0.moe_router.proj.weight"].shape == (_M_N_EXPERTS, _M_H)

    def test_router_scale_and_per_expert_scale_pass_through(self, tmp_path: Path) -> None:
        tensors = _make_moe_tensor_map()
        yielded = dict(_run_iter_with_fakes(tensors, _iter_moe_transformer, tmp_path, list(tensors), _M_LAYERS))
        np.testing.assert_array_equal(yielded["model.layers.0.moe_router.scale"], tensors["layer_0.mlp.router_scale"])
        np.testing.assert_array_equal(
            yielded["model.layers.0.moe_router.per_expert_scale"], tensors["layer_0.mlp.per_expert_scale"]
        )

    def test_experts_packed_gate_up_reshape(self, tmp_path: Path) -> None:
        """[E, 2, I_moe, H] → [E, 2·I_moe, H] preserving gate-then-up order on axis 1."""
        tensors = _make_moe_tensor_map()
        yielded = dict(_run_iter_with_fakes(tensors, _iter_moe_transformer, tmp_path, list(tensors), _M_LAYERS))
        src = tensors["layer_0.mlp.gating_einsum.w"]  # (E, 2, I_moe, H)
        emitted = yielded["model.layers.0.moe_experts.gate_up_proj"]
        assert emitted.shape == (_M_N_EXPERTS, 2 * _M_I_MOE, _M_H)
        # Row 0..I_moe on axis 1 should be the gate half; I_moe..2·I_moe should be up.
        np.testing.assert_array_equal(emitted[:, :_M_I_MOE, :], src[:, 0, :, :])
        np.testing.assert_array_equal(emitted[:, _M_I_MOE:, :], src[:, 1, :, :])

    def test_experts_down_transpose(self, tmp_path: Path) -> None:
        """[E, I_moe, H] → [E, H, I_moe] (transpose last two axes)."""
        tensors = _make_moe_tensor_map()
        yielded = dict(_run_iter_with_fakes(tensors, _iter_moe_transformer, tmp_path, list(tensors), _M_LAYERS))
        src = tensors["layer_0.mlp.linear.w"]  # (E, I_moe, H)
        emitted = yielded["model.layers.0.moe_experts.down_proj"]
        assert emitted.shape == (_M_N_EXPERTS, _M_H, _M_I_MOE)
        np.testing.assert_array_equal(emitted, src.transpose(0, 2, 1))

    def test_moe_norms_all_six_emitted(self, tmp_path: Path) -> None:
        tensors = _make_moe_tensor_map()
        yielded = dict(_run_iter_with_fakes(tensors, _iter_moe_transformer, tmp_path, list(tensors), _M_LAYERS))
        norm_map = {
            "pre_feedforward_layernorm.weight": "pre_ffw_norm.scale",
            "post_feedforward_layernorm_1.weight": "post_ffw1_norm.scale",
            "pre_feedforward_layernorm_2.weight": "pre_ffw2_norm.scale",
            "post_feedforward_layernorm_2.weight": "post_ffw2_norm.scale",
            "post_feedforward_layernorm.weight": "post_ffw_norm.scale",
        }
        for hf_suffix, orbax_suffix in norm_map.items():
            np.testing.assert_array_equal(yielded[f"model.layers.0.{hf_suffix}"], tensors[f"layer_0.{orbax_suffix}"])

    def test_skip_scale_preserved_verbatim(self, tmp_path: Path) -> None:
        tensors = _make_moe_tensor_map()
        yielded = dict(_run_iter_with_fakes(tensors, _iter_moe_transformer, tmp_path, list(tensors), _M_LAYERS))
        emitted = yielded["model.layers.0.moe_skip_scale.weight"]
        np.testing.assert_array_equal(emitted, tensors["layer_0.skip_scale"])
        assert emitted.shape == (1,)

    def test_optional_post_ffw_norm_omitted_if_absent(self, tmp_path: Path) -> None:
        """If Orbax lacks post_ffw_norm.scale, the safetensors key is simply not emitted."""
        tensors = _make_moe_tensor_map()
        del tensors["layer_0.post_ffw_norm.scale"]
        del tensors["layer_1.post_ffw_norm.scale"]
        yielded = dict(_run_iter_with_fakes(tensors, _iter_moe_transformer, tmp_path, list(tensors), _M_LAYERS))
        assert "model.layers.0.post_feedforward_layernorm.weight" not in yielded


class TestAttnLayerKVLayouts:
    """`_iter_attn_layer` handles sliding (kv_einsum) + full (k_einsum only) + split (k+v) layouts."""

    def _build_layer(self, kv_layout: str) -> dict[str, np.ndarray]:
        """Build a one-layer attention fixture in the requested layout."""
        rng = np.random.default_rng(seed=4)
        tensors: dict[str, np.ndarray] = {
            "layer_0.attn.q_einsum.w": rng.standard_normal((4, 16, 8)).astype(np.float32),
            "layer_0.attn.attn_vec_einsum.w": rng.standard_normal((4, 8, 16)).astype(np.float32),
            "layer_0.attn.query_norm.scale": rng.standard_normal((8,)).astype(np.float32),
            "layer_0.attn.key_norm.scale": rng.standard_normal((8,)).astype(np.float32),
        }
        if kv_layout == "combined":
            tensors["layer_0.attn.kv_einsum.w"] = rng.standard_normal((2, 2, 16, 8)).astype(np.float32)
        elif kv_layout == "split":
            tensors["layer_0.attn.k_einsum.w"] = rng.standard_normal((2, 16, 8)).astype(np.float32)
            tensors["layer_0.attn.v_einsum.w"] = rng.standard_normal((2, 16, 8)).astype(np.float32)
        elif kv_layout == "k_only":
            tensors["layer_0.attn.k_einsum.w"] = rng.standard_normal((2, 16, 8)).astype(np.float32)
        return tensors

    def test_combined_kv_einsum_splits(self, tmp_path: Path) -> None:
        from mogemma.convert import _iter_attn_layer

        tensors = self._build_layer("combined")
        keys = set(tensors)

        def fake_open(_path: object, name: str) -> np.ndarray:
            return tensors[name]

        with patch.object(OrbaxLoader, "open_tensor", staticmethod(fake_open)):
            yielded = dict(_iter_attn_layer(tmp_path, keys, 0))
        kv = tensors["layer_0.attn.kv_einsum.w"]
        assert not np.array_equal(
            yielded["model.layers.0.self_attn.k_proj.weight"], yielded["model.layers.0.self_attn.v_proj.weight"]
        )
        np.testing.assert_array_equal(
            yielded["model.layers.0.self_attn.k_proj.weight"], kv[0].transpose(0, 2, 1).reshape(-1, 16)
        )

    def test_split_k_and_v(self, tmp_path: Path) -> None:
        from mogemma.convert import _iter_attn_layer

        tensors = self._build_layer("split")
        keys = set(tensors)

        def fake_open(_path: object, name: str) -> np.ndarray:
            return tensors[name]

        with patch.object(OrbaxLoader, "open_tensor", staticmethod(fake_open)):
            yielded = dict(_iter_attn_layer(tmp_path, keys, 0))
        assert not np.array_equal(
            yielded["model.layers.0.self_attn.k_proj.weight"], yielded["model.layers.0.self_attn.v_proj.weight"]
        )

    def test_k_only_shares_with_v(self, tmp_path: Path) -> None:
        """Full-attention layers with KV-sharing: v_proj == k_proj."""
        from mogemma.convert import _iter_attn_layer

        tensors = self._build_layer("k_only")
        keys = set(tensors)

        def fake_open(_path: object, name: str) -> np.ndarray:
            return tensors[name]

        with patch.object(OrbaxLoader, "open_tensor", staticmethod(fake_open)):
            yielded = dict(_iter_attn_layer(tmp_path, keys, 0))
        np.testing.assert_array_equal(
            yielded["model.layers.0.self_attn.k_proj.weight"], yielded["model.layers.0.self_attn.v_proj.weight"]
        )

    def test_missing_both_raises(self, tmp_path: Path) -> None:
        from mogemma.convert import _iter_attn_layer

        tensors = self._build_layer("combined")
        del tensors["layer_0.attn.kv_einsum.w"]

        def fake_open(_path: object, name: str) -> np.ndarray:
            return tensors[name]

        with (
            patch.object(OrbaxLoader, "open_tensor", staticmethod(fake_open)),
            pytest.raises(KeyError, match="kv_einsum"),
        ):
            list(_iter_attn_layer(tmp_path, set(tensors), 0))


# Vision shrunk shapes (E2B-it uses L=16, H_v=768, n_heads=12, head_dim=64, intermediate=3072).
_VL, _VH, _V_HEADS, _V_KV_HEADS, _V_HEAD_DIM, _V_INTER = 3, 48, 4, 4, 12, 192


def _make_vision_tensor_map(num_layers: int = _VL) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed=2)
    pfx = "vision_encoder.transformer.stacked_layers.block"
    return {
        "vision_encoder.entry.input_projection.w": rng.standard_normal((_VH, _VH)).astype(np.float32),
        "vision_encoder.entry.pos_emb": rng.standard_normal((10240, 2, _VH)).astype(np.float32),
        "embedder.mm_input_projection.w": rng.standard_normal((_VH, 96)).astype(np.float32),
        f"{pfx}.attn.q_einsum.w": rng.standard_normal((num_layers, _V_HEADS, _VH, _V_HEAD_DIM)).astype(np.float32),
        f"{pfx}.attn.kv_einsum.w": rng.standard_normal((num_layers, 2, _V_KV_HEADS, _VH, _V_HEAD_DIM)).astype(
            np.float32
        ),
        f"{pfx}.attn.attn_vec_einsum.w": rng.standard_normal((num_layers, _V_HEADS, _V_HEAD_DIM, _VH)).astype(
            np.float32
        ),
        f"{pfx}.mlp.gating_einsum.w": rng.standard_normal((num_layers, 2, _V_INTER, _VH)).astype(np.float32),
        f"{pfx}.mlp.linear.w": rng.standard_normal((num_layers, _V_INTER, _VH)).astype(np.float32),
        f"{pfx}.pre_attention_norm.scale": rng.standard_normal((num_layers, _VH)).astype(np.float32),
        f"{pfx}.pre_ffw_norm.scale": rng.standard_normal((num_layers, _VH)).astype(np.float32),
        f"{pfx}.post_ffw_norm.scale": rng.standard_normal((num_layers, _VH)).astype(np.float32),
    }


class TestVisionIterator:
    """`_iter_vision` emits the vision tensors consumed by forward_vision_layer (GEGLU)."""

    def test_yields_expected_name_set(self, tmp_path: Path) -> None:
        tensors = _make_vision_tensor_map()
        yielded = _run_iter_with_fakes(tensors, _iter_vision, tmp_path, _VL)
        names = {name for name, _ in yielded}

        expected = {
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "vision_tower.vision_model.post_layernorm.weight",
            "multi_modal_projector.linear.weight",
        }
        for i in range(_VL):
            pfx = f"vision_tower.vision_model.encoder.layers.{i}"
            expected |= {
                f"{pfx}.self_attn.q_proj.weight",
                f"{pfx}.self_attn.k_proj.weight",
                f"{pfx}.self_attn.v_proj.weight",
                f"{pfx}.self_attn.out_proj.weight",
                f"{pfx}.mlp.fc1.weight",
                f"{pfx}.mlp.fc1_up.weight",
                f"{pfx}.mlp.fc2.weight",
                f"{pfx}.layer_norm1.weight",
                f"{pfx}.layer_norm2.weight",
            }
        assert names == expected

    def test_geglu_gate_and_up_are_distinct_halves(self, tmp_path: Path) -> None:
        """fc1 ← gating[0] (gate), fc1_up ← gating[1] (up). They must differ."""
        tensors = _make_vision_tensor_map()
        yielded = dict(_run_iter_with_fakes(tensors, _iter_vision, tmp_path, _VL))
        pfx = "vision_encoder.transformer.stacked_layers.block"
        src = tensors[f"{pfx}.mlp.gating_einsum.w"]  # (L, 2, I, H)
        np.testing.assert_array_equal(yielded["vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight"], src[0, 0])
        np.testing.assert_array_equal(
            yielded["vision_tower.vision_model.encoder.layers.0.mlp.fc1_up.weight"], src[0, 1]
        )
        assert not np.array_equal(src[0, 0], src[0, 1])

    def test_fc2_transposes_orbax_linear(self, tmp_path: Path) -> None:
        tensors = _make_vision_tensor_map()
        yielded = dict(_run_iter_with_fakes(tensors, _iter_vision, tmp_path, _VL))
        pfx = "vision_encoder.transformer.stacked_layers.block"
        src = tensors[f"{pfx}.mlp.linear.w"]
        emitted = yielded["vision_tower.vision_model.encoder.layers.0.mlp.fc2.weight"]
        np.testing.assert_array_equal(emitted, src[0].T)
        assert emitted.shape == (_VH, _V_INTER)

    def test_post_layernorm_uses_last_layer_post_ffw(self, tmp_path: Path) -> None:
        tensors = _make_vision_tensor_map()
        yielded = dict(_run_iter_with_fakes(tensors, _iter_vision, tmp_path, _VL))
        pfx = "vision_encoder.transformer.stacked_layers.block"
        expected = tensors[f"{pfx}.post_ffw_norm.scale"][-1]
        np.testing.assert_array_equal(yielded["vision_tower.vision_model.post_layernorm.weight"], expected)


from mogemma.convert import _generate_config_json


class TestGenerateConfigJson:
    """`_generate_config_json` synthesizes a HF-compatible config.json from Orbax tensor shapes."""

    def _fake_shape_oracle(self, shapes: dict[str, tuple[int, ...]]) -> Callable[[str], tuple[int, ...]]:
        """Build a `(name)->shape` callable that simulates OrbaxLoader shape access without I/O."""

        def _oracle(name: str) -> tuple[int, ...]:
            return shapes[name]

        return _oracle

    def test_base_variant_emits_required_fields(self) -> None:
        shapes = {
            "embedder.input_embedding": (262144, 1536),
            "layer_0.attn.q_einsum.w": (8, 1536, 256),
            "layer_0.attn.kv_einsum.w": (2, 1, 1536, 256),
            "layer_0.mlp.gating_einsum.w": (2, 6144, 1536),
        }
        keys = list(shapes) + [f"layer_{i}.attn.q_einsum.w" for i in range(1, 35)]
        config = _generate_config_json(keys, self._fake_shape_oracle(shapes))

        assert str(config["model_type"]).startswith("gemma4")
        assert config["num_hidden_layers"] == 35
        assert config["hidden_size"] == 1536
        assert config["vocab_size"] == 262144
        assert config["num_attention_heads"] == 8
        assert config["num_key_value_heads"] == 1
        assert config["head_dim"] == 256
        assert config["intermediate_size"] == 6144

    def test_base_variant_is_dense_text(self) -> None:
        shapes = {
            "embedder.input_embedding": (262144, 1536),
            "layer_0.attn.q_einsum.w": (8, 1536, 256),
            "layer_0.attn.kv_einsum.w": (2, 1, 1536, 256),
            "layer_0.mlp.gating_einsum.w": (2, 6144, 1536),
        }
        keys = list(shapes)
        config = _generate_config_json(keys, self._fake_shape_oracle(shapes))
        assert "hidden_size_per_layer_input" not in config
        assert "num_local_experts" not in config

    def test_ple_variant_includes_per_layer_fields(self) -> None:
        shapes = {
            "embedder.input_embedding": (262144, 1536),
            "embedder.per_layer_embeddings": (262144, 35, 256),
            "layer_0.attn.q_einsum.w": (8, 1536, 256),
            "layer_0.attn.kv_einsum.w": (2, 1, 1536, 256),
            "layer_0.mlp.gating_einsum.w": (2, 6144, 1536),
        }
        keys = list(shapes) + [f"layer_{i}.attn.q_einsum.w" for i in range(1, 35)]
        config = _generate_config_json(keys, self._fake_shape_oracle(shapes))

        assert config["hidden_size_per_layer_input"] == 256
        assert config["vocab_size_per_layer_input"] == 262144

    def test_moe_variant_includes_expert_fields(self) -> None:
        shapes = {
            "embedder.input_embedding": (262144, 3072),
            "layer_0.attn.q_einsum.w": (16, 3072, 128),
            "layer_0.attn.kv_einsum.w": (2, 2, 3072, 128),
            "layer_0.mlp.router_logits.w": (3072, 128),
            "layer_0.mlp.gating_einsum.w": (128, 2, 704, 3072),
        }
        keys = list(shapes) + [f"layer_{i}.attn.q_einsum.w" for i in range(1, 40)]
        config = _generate_config_json(keys, self._fake_shape_oracle(shapes))

        assert config["num_local_experts"] == 128
        assert config["moe_intermediate_size"] == 704

    def test_passes_project_validate_config_json(self) -> None:
        """Generated config must satisfy `HubManager.validate_config_json`."""
        from mogemma.hub import HubManager

        shapes = {
            "embedder.input_embedding": (262144, 1536),
            "layer_0.attn.q_einsum.w": (8, 1536, 256),
            "layer_0.attn.kv_einsum.w": (2, 1, 1536, 256),
            "layer_0.mlp.gating_einsum.w": (2, 6144, 1536),
        }
        keys = list(shapes)
        config = _generate_config_json(keys, self._fake_shape_oracle(shapes))
        HubManager.validate_config_json(config)  # must not raise


class TestConvertOrbaxToSafetensorsRoundTrip:
    """End-to-end: synthetic Orbax → convert → SafetensorsLoader reads it back."""

    def _base_contract_names(self, num_layers: int) -> set[str]:
        names = {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"}
        for n in range(num_layers):
            pfx = f"model.layers.{n}"
            names |= {
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
        return names

    def _install_fake_orbax(self, tensors: dict[str, np.ndarray]) -> tuple[Any, Any]:
        """Return (enumerate_patch, open_patch) that back the fake Orbax inventory.

        Typed as ``Any`` because ``unittest.mock._patch`` is not exported; the
        returned objects are genuine context managers at runtime.
        """

        def fake_enumerate(_path: object) -> list[str]:
            return list(tensors)

        def fake_open(_path: object, name: str) -> np.ndarray:
            return tensors[name]

        return (
            patch.object(OrbaxLoader, "enumerate_tensors", staticmethod(fake_enumerate)),
            patch.object(OrbaxLoader, "open_tensor", staticmethod(fake_open)),
        )

    def test_writes_single_shard_and_config_json(self, tmp_path: Path) -> None:
        from mogemma.convert import convert_orbax_to_safetensors

        tensors = _make_base_tensor_map()
        enum_patch, open_patch = self._install_fake_orbax(tensors)
        with enum_patch, open_patch:
            written = convert_orbax_to_safetensors(tmp_path)

        assert written == [tmp_path / "model.safetensors"]
        assert (tmp_path / "model.safetensors").exists()
        assert (tmp_path / "config.json").exists()

        config = json.loads((tmp_path / "config.json").read_text())
        assert str(config["model_type"]).startswith("gemma4")
        assert config["num_hidden_layers"] == _N_LAYERS
        assert config["hidden_size"] == _H

    def test_safetensors_loader_can_load_output(self, tmp_path: Path) -> None:
        from mogemma.convert import convert_orbax_to_safetensors
        from mogemma.loader import SafetensorsLoader, auto_loader

        tensors = _make_base_tensor_map()
        enum_patch, open_patch = self._install_fake_orbax(tensors)
        with enum_patch, open_patch:
            convert_orbax_to_safetensors(tmp_path)

        assert SafetensorsLoader.can_load(tmp_path) is True

        loader = auto_loader(tmp_path)
        try:
            assert isinstance(loader, SafetensorsLoader)
            metadata = loader.get_tensor_metadata()
            emitted = set(metadata.keys())
            expected = self._base_contract_names(_N_LAYERS)
            missing = expected - emitted
            assert not missing, f"Contract tensors missing from safetensors output: {missing}"
        finally:
            loader.close()

    def test_tensor_values_survive_round_trip(self, tmp_path: Path) -> None:
        """Round-trip preserves transformed values for a representative tensor."""
        from safetensors import safe_open  # type: ignore[import-untyped]

        from mogemma.convert import convert_orbax_to_safetensors

        tensors = _make_base_tensor_map()
        enum_patch, open_patch = self._install_fake_orbax(tensors)
        with enum_patch, open_patch:
            convert_orbax_to_safetensors(tmp_path)

        with safe_open(str(tmp_path / "model.safetensors"), framework="numpy") as f:  # type: ignore[no-untyped-call]
            gate = f.get_tensor("model.layers.0.mlp.gate_proj.weight")
            up = f.get_tensor("model.layers.0.mlp.up_proj.weight")

        src = tensors["layer_0.mlp.gating_einsum.w"]
        np.testing.assert_array_equal(gate, src[0])
        np.testing.assert_array_equal(up, src[1])

    def test_preserves_existing_config_json(self, tmp_path: Path) -> None:
        """If an Orbax checkpoint shipped a config.json, conversion must not overwrite it."""
        from mogemma.convert import convert_orbax_to_safetensors

        preset = {"model_type": "gemma4_text", "custom_marker": 42}
        (tmp_path / "config.json").write_text(json.dumps(preset))

        tensors = _make_base_tensor_map()
        enum_patch, open_patch = self._install_fake_orbax(tensors)
        with enum_patch, open_patch:
            convert_orbax_to_safetensors(tmp_path)

        after = json.loads((tmp_path / "config.json").read_text())
        assert after == preset

    def test_raises_when_no_orbax_keys_present(self, tmp_path: Path) -> None:
        from mogemma.convert import convert_orbax_to_safetensors

        def fake_enumerate(_path: object) -> list[str]:
            return []

        with (
            patch.object(OrbaxLoader, "enumerate_tensors", staticmethod(fake_enumerate)),
            pytest.raises(ValueError, match="No Orbax tensors"),
        ):
            convert_orbax_to_safetensors(tmp_path)
