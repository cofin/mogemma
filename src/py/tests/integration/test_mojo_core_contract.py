"""Python-to-Mojo Gemma 4 runtime contract tests."""

from __future__ import annotations

import importlib.util
from typing import Protocol, cast

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("mogemma._core") is None, reason="mogemma._core is not built")


class _CoreInitProtocol(Protocol):
    def init_model_with_options(
        self,
        metadata: dict[str, tuple[int, tuple[int, ...], str]],
        architecture_overrides: dict[str, object],
        device_selection: dict[str, object],
    ) -> object: ...


class _CoreAudioProtocol(Protocol):
    def process_audio(self, llm: object, features: object, num_tokens: int) -> object: ...


def _meta(arrays: dict[str, np.ndarray]) -> dict[str, tuple[int, tuple[int, ...], str]]:
    return {name: (int(array.ctypes.data), tuple(array.shape), str(array.dtype)) for name, array in arrays.items()}


def _tiny_dense_arrays() -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": np.ones((8, 4), dtype=np.float32),
        "model.norm.weight": np.ones((4,), dtype=np.float32),
        "lm_head.weight": np.ones((8, 4), dtype=np.float32),
    }
    prefix = "model.layers.0"
    arrays.update({
        f"{prefix}.input_layernorm.weight": np.ones((4,), dtype=np.float32),
        f"{prefix}.post_attention_layernorm.weight": np.ones((4,), dtype=np.float32),
        f"{prefix}.self_attn.q_proj.weight": np.ones((4, 4), dtype=np.float32),
        f"{prefix}.self_attn.k_proj.weight": np.ones((2, 4), dtype=np.float32),
        f"{prefix}.self_attn.v_proj.weight": np.ones((2, 4), dtype=np.float32),
        f"{prefix}.self_attn.o_proj.weight": np.ones((4, 4), dtype=np.float32),
        f"{prefix}.mlp.gate_proj.weight": np.ones((6, 4), dtype=np.float32),
        f"{prefix}.mlp.up_proj.weight": np.ones((6, 4), dtype=np.float32),
        f"{prefix}.mlp.down_proj.weight": np.ones((4, 6), dtype=np.float32),
        f"{prefix}.self_attn.q_norm.weight": np.ones((2,), dtype=np.float32),
        f"{prefix}.self_attn.k_norm.weight": np.ones((2,), dtype=np.float32),
        f"{prefix}.pre_feedforward_layernorm.weight": np.ones((4,), dtype=np.float32),
        f"{prefix}.post_feedforward_layernorm.weight": np.ones((4,), dtype=np.float32),
    })
    return arrays


def test_mojo_core_init_gemma4_dense_contract() -> None:
    from mogemma import _core

    core = cast("_CoreInitProtocol", _core)
    arrays = _tiny_dense_arrays()
    llm = cast(
        "dict[str, object]",
        core.init_model_with_options(
            _meta(arrays),
            {"window_size": 8, "max_seq_len": 16, "layer_types": [0]},
            {
                "backend": "cpu",
                "device_kind": "cpu",
                "device_index": None,
                "requested": "cpu",
                "availability_source": "test",
                "strict": False,
            },
        ),
    )

    assert llm["arch"] == "gemma4"
    assert llm["num_layers"] == 1
    assert llm["hidden_size"] == 4
    assert llm["head_dim"] == 2
    assert llm["num_heads"] == 2
    assert llm["num_kv_heads"] == 1
    assert llm["window_size"] == 8
    assert llm["max_seq_len"] == 16
    assert llm["device_backend"] == "cpu"
    assert llm["_gpu_initialized"] == 0
    assert cast("int", llm["_arena_ptr"]) != 0
    assert cast("int", llm["_kv_cache_ptr"]) != 0


def test_mojo_process_audio_rejects_instead_of_appending_zero_embeddings() -> None:
    from mogemma import _core

    core = cast("_CoreAudioProtocol", _core)
    llm = {
        "hidden_size": 4,
        "audio_embeddings": [],
        "audio_hidden_size": 640,
        "audio_num_heads": 4,
        "audio_intermediate_size": 1280,
        "audio_n_mels": 80,
    }
    features = np.zeros((80, 2), dtype=np.float32)

    with pytest.raises(Exception, match="Audio input is recognized but not implemented"):
        core.process_audio(llm, features, 2)
    assert llm["audio_embeddings"] == []
