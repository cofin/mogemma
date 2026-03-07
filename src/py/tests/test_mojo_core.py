import numpy as np
import numpy.typing as npt
import pytest

_core = pytest.importorskip("mogemma._core", exc_type=ImportError)

_EXPECTED_HEAD_DIM = 2
_EXPECTED_HIDDEN_SIZE = 4
_EXPECTED_VOCAB_SIZE = 10
_EXPECTED_PER_LAYER_DIM_SMALL = 2
_EXPECTED_PER_LAYER_DIM_LARGE = 256


def _get_ptr(arr: npt.NDArray[np.float32]) -> int:
    return int(arr.__array_interface__["data"][0])


def test_mojo_core_init_standard() -> None:
    # Allocate some real arrays
    tensors = {
        "model.embed_tokens.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "lm_head.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.zeros((8, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.v_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.o_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 8), dtype=np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.up_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.down_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 16), dtype=np.float32),
        "model.layers.0.self_attn.q_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.self_attn.k_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.pre_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
    }

    # Mock metadata
    metadata = {k: (_get_ptr(v), v.shape) for k, v in tensors.items()}

    llm = _core.init_model(metadata)
    assert llm["arch"] == "standard"
    assert llm["num_layers"] == 1
    assert llm["head_dim"] == _EXPECTED_HEAD_DIM
    expected_num_heads = 4
    assert llm["num_heads"] == expected_num_heads
    assert llm["num_kv_heads"] == _EXPECTED_HEAD_DIM
    assert llm["hidden_size"] == _EXPECTED_HIDDEN_SIZE
    assert llm["vocab_size"] == _EXPECTED_VOCAB_SIZE
    expected_session_kv_cache_len = llm["num_layers"] * llm["max_seq_len"] * llm["num_kv_heads"] * llm["head_dim"]
    assert llm["session_kv_cache_len"] == expected_session_kv_cache_len
    assert llm["step_scratch_len"] == llm["hidden_size"] * 160
    assert llm["embedding_scratch_len"] == llm["hidden_size"] * 180
    assert llm.get("descriptor_build_count", 1) == 1


def test_mojo_core_init_model_with_options_caches_runtime_device_descriptor() -> None:
    if not hasattr(_core, "init_model_with_options"):
        pytest.skip("init_model_with_options is unavailable in current compiled mogemma._core")

    tensors = {
        "model.embed_tokens.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "lm_head.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.zeros((8, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.v_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.o_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 8), dtype=np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.up_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.down_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 16), dtype=np.float32),
        "model.layers.0.self_attn.q_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.self_attn.k_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.pre_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
    }
    metadata = {k: (_get_ptr(v), v.shape) for k, v in tensors.items()}

    llm = _core.init_model_with_options(
        metadata,
        {"head_dim": 128, "rope_base": 1_000_000.0},
        {
            "requested": "gpu:2",
            "backend": "gpu",
            "device_kind": "gpu",
            "device_index": 2,
            "strict": True,
            "availability_source": "override",
        },
    )

    assert llm["architecture_overrides"] == {"head_dim": 128, "rope_base": 1_000_000.0}
    assert llm["device_selection"] == {
        "requested": "gpu:2",
        "backend": "gpu",
        "device_kind": "gpu",
        "device_index": 2,
        "strict": True,
        "availability_source": "override",
    }
    assert llm["device_backend"] == "gpu"
    assert llm["device_kind"] == "gpu"
    expected_device_index = 2
    assert llm["device_index"] == expected_device_index
    assert llm["device_request"] == "gpu:2"
    assert llm["device_availability_source"] == "override"
    assert llm["device_strict"] is True


def test_mojo_core_step_standard() -> None:
    tensors = {
        "model.embed_tokens.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "lm_head.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.zeros((8, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.v_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.o_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 8), dtype=np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.up_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.down_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 16), dtype=np.float32),
        "model.layers.0.self_attn.q_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.self_attn.k_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.pre_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
    }
    metadata = {k: (_get_ptr(v), v.shape) for k, v in tensors.items()}
    llm = _core.init_model(metadata)

    logits = _core.step(llm, 1, 0.0, 0, 0.0)
    assert logits.shape == (_EXPECTED_VOCAB_SIZE,)
    assert llm["pos"] == 1
    assert llm.get("descriptor_build_count", 1) == 1
    assert llm.get("step_backend") == "cpu"
    assert llm.get("fallback_reason") == "requested"


def test_mojo_core_step_standard_cuda() -> None:
    if not hasattr(_core, "init_model_with_options"):
        pytest.skip("init_model_with_options is unavailable")

    tensors = {
        "model.embed_tokens.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "lm_head.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.zeros((8, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.v_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.o_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 8), dtype=np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.up_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.down_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 16), dtype=np.float32),
        "model.layers.0.self_attn.q_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.self_attn.k_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.pre_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
    }
    metadata = {k: (_get_ptr(v), v.shape) for k, v in tensors.items()}
    llm = _core.init_model_with_options(
        metadata,
        {},
        {"backend": "cuda", "device_kind": "gpu", "requested": "cuda", "strict": False, "availability_source": "auto"},
    )

    logits = _core.step(llm, 1, 0.0, 0, 0.0)
    assert logits.shape == (_EXPECTED_VOCAB_SIZE,)
    assert llm.get("step_backend") == "cuda"
    assert llm.get("fallback_reason") == "none"
    assert llm.get("debug_launch_count") == 1

    # second token uses same latch
    logits = _core.step(llm, 2, 0.0, 0, 0.0)
    assert llm.get("step_backend") == "cuda"
    expected_launch_count = 2
    assert llm.get("debug_launch_count") == expected_launch_count


def test_mojo_core_step_standard_cuda_zero_alloc_validation() -> None:
    if not hasattr(_core, "init_model_with_options"):
        pytest.skip("init_model_with_options is unavailable")

    tensors = {
        "model.embed_tokens.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "lm_head.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.zeros((8, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.v_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.o_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 8), dtype=np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.up_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.down_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 16), dtype=np.float32),
        "model.layers.0.self_attn.q_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.self_attn.k_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.pre_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
    }
    metadata = {k: (_get_ptr(v), v.shape) for k, v in tensors.items()}
    llm = _core.init_model_with_options(
        metadata,
        {},
        {"backend": "cuda", "device_kind": "gpu", "requested": "cuda", "strict": False, "availability_source": "auto"},
    )

    # First step
    logits1 = _core.step(llm, 1, 0.0, 0, 0.0)
    assert logits1.shape == (_EXPECTED_VOCAB_SIZE,)
    assert llm.get("step_backend") == "cuda"

    # Second step tests cache writing / pointer reuse without blowing up
    _ = _core.step(llm, 2, 0.0, 0, 0.0)
    assert llm.get("step_backend") == "cuda"
    expected_pos = 2
    assert llm["pos"] == expected_pos


def test_mojo_core_embeddings_standard_uses_local_rope_when_sequence_exceeds_runtime_window() -> None:
    tensors = {
        "model.embed_tokens.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "lm_head.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.zeros((8, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.v_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.o_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 8), dtype=np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.up_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.down_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 16), dtype=np.float32),
        "model.layers.0.self_attn.q_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.self_attn.k_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.pre_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
    }
    metadata = {k: (_get_ptr(v), v.shape) for k, v in tensors.items()}
    llm = _core.init_model(metadata)

    # Force local RoPE path in embedding call.
    llm["max_seq_len"] = 1
    embeddings = _core.generate_embeddings(llm, [[1, 2, 3]])

    assert embeddings.shape == (1, _EXPECTED_HIDDEN_SIZE)


def test_mojo_core_init_nano() -> None:
    tensors = {
        "model.embed_tokens.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "lm_head.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.per_layer_embed.weight": np.zeros(
            (_EXPECTED_VOCAB_SIZE, 30, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.per_layer_embed.projection.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, 30, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.per_layer_embed.norm.weight": np.zeros((_EXPECTED_PER_LAYER_DIM_SMALL,), dtype=np.float32),
        "model.layers.0.altup.router.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.altup.router_norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.altup.prediction_coefs": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.altup.correction_coefs": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.altup.output_scale": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.laurel.down_proj.weight": np.zeros(
            (_EXPECTED_PER_LAYER_DIM_SMALL, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.laurel.up_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.layers.0.laurel.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.per_layer_map.gate.weight": np.zeros(
            (_EXPECTED_PER_LAYER_DIM_SMALL, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.per_layer_map.projection.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.layers.0.per_layer_map.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.zeros((8, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.v_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.o_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 8), dtype=np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.up_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.down_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 16), dtype=np.float32),
        "model.layers.0.self_attn.q_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.self_attn.k_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.pre_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
    }

    metadata = {k: (_get_ptr(v), v.shape) for k, v in tensors.items()}

    llm = _core.init_model(metadata)
    assert llm["arch"] == "nano"
    assert llm["num_layers"] == 1
    assert llm["head_dim"] == _EXPECTED_HEAD_DIM
    expected_num_heads = 4
    assert llm["num_heads"] == expected_num_heads
    assert llm["per_layer_dim"] == _EXPECTED_PER_LAYER_DIM_SMALL


def test_mojo_core_init_nano_with_3d_per_layer_embed() -> None:
    tensors = {
        "model.embed_tokens.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "lm_head.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.per_layer_embed.weight": np.zeros(
            (_EXPECTED_VOCAB_SIZE, 30, _EXPECTED_PER_LAYER_DIM_LARGE), dtype=np.float32
        ),
        "model.per_layer_embed.projection.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, 30, _EXPECTED_PER_LAYER_DIM_LARGE), dtype=np.float32
        ),
        "model.per_layer_embed.norm.weight": np.zeros((_EXPECTED_PER_LAYER_DIM_LARGE,), dtype=np.float32),
        "model.layers.0.altup.router.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.altup.router_norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.altup.prediction_coefs": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.altup.correction_coefs": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.altup.output_scale": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.laurel.down_proj.weight": np.zeros(
            (_EXPECTED_PER_LAYER_DIM_SMALL, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.laurel.up_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.layers.0.laurel.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.per_layer_map.gate.weight": np.zeros(
            (_EXPECTED_PER_LAYER_DIM_LARGE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.per_layer_map.projection.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_PER_LAYER_DIM_LARGE), dtype=np.float32
        ),
        "model.layers.0.per_layer_map.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.zeros((8, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.v_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.o_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 8), dtype=np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.up_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.down_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 16), dtype=np.float32),
        "model.layers.0.self_attn.q_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.self_attn.k_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.pre_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
    }
    metadata = {k: (_get_ptr(v), v.shape) for k, v in tensors.items()}

    llm = _core.init_model(metadata)
    assert llm["arch"] == "nano"
    assert llm["per_layer_dim"] == _EXPECTED_PER_LAYER_DIM_LARGE


def test_mojo_core_step_nano() -> None:
    tensors = {
        "model.embed_tokens.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "lm_head.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.per_layer_embed.weight": np.zeros(
            (_EXPECTED_VOCAB_SIZE, 30, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.per_layer_embed.projection.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, 30, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.per_layer_embed.norm.weight": np.zeros((_EXPECTED_PER_LAYER_DIM_SMALL,), dtype=np.float32),
        "model.layers.0.altup.router.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.altup.router_norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.altup.prediction_coefs": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.altup.correction_coefs": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.altup.output_scale": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.laurel.down_proj.weight": np.zeros(
            (_EXPECTED_PER_LAYER_DIM_SMALL, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.laurel.up_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.layers.0.laurel.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.per_layer_map.gate.weight": np.zeros(
            (_EXPECTED_PER_LAYER_DIM_SMALL, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.per_layer_map.projection.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.layers.0.per_layer_map.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.zeros((8, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.v_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.o_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 8), dtype=np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.up_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.down_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 16), dtype=np.float32),
        "model.layers.0.self_attn.q_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.self_attn.k_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.pre_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.altup.projection.0.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.altup.projection.1.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.altup.projection.2.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.altup.unembed.0.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.altup.unembed.1.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.altup.unembed.2.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
    }
    metadata = {k: (_get_ptr(v), v.shape) for k, v in tensors.items()}
    llm = _core.init_model(metadata)

    logits = _core.step(llm, 1, 0.0, 0, 0.0)
    assert logits.shape == (_EXPECTED_VOCAB_SIZE,)
    assert llm["pos"] == 1


def test_mojo_core_detects_nano_kv_share_start_boundary() -> None:
    tensors = {
        "model.embed_tokens.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "lm_head.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.per_layer_embed.weight": np.zeros(
            (_EXPECTED_VOCAB_SIZE, 30, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.per_layer_embed.projection.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, 30, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.per_layer_embed.norm.weight": np.zeros((_EXPECTED_PER_LAYER_DIM_SMALL,), dtype=np.float32),
    }

    for layer_idx in range(2):
        pfx = f"model.layers.{layer_idx}"
        tensors[f"{pfx}.altup.router.weight"] = np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        )
        tensors[f"{pfx}.altup.router_norm.weight"] = np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32)
        tensors[f"{pfx}.altup.prediction_coefs"] = np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        )
        tensors[f"{pfx}.altup.correction_coefs"] = np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        )
        tensors[f"{pfx}.altup.output_scale"] = np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32)
        tensors[f"{pfx}.laurel.down_proj.weight"] = np.zeros(
            (_EXPECTED_PER_LAYER_DIM_SMALL, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        )
        tensors[f"{pfx}.laurel.up_proj.weight"] = np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        )
        tensors[f"{pfx}.laurel.norm.weight"] = np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32)
        tensors[f"{pfx}.per_layer_map.gate.weight"] = np.zeros(
            (_EXPECTED_PER_LAYER_DIM_SMALL, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        )
        tensors[f"{pfx}.per_layer_map.projection.weight"] = np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        )
        tensors[f"{pfx}.per_layer_map.norm.weight"] = np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32)
        tensors[f"{pfx}.input_layernorm.weight"] = np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32)
        tensors[f"{pfx}.post_attention_layernorm.weight"] = np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32)
        tensors[f"{pfx}.self_attn.q_proj.weight"] = np.zeros((8, _EXPECTED_HIDDEN_SIZE), dtype=np.float32)

        if layer_idx == 0:
            tensors[f"{pfx}.self_attn.k_proj.weight"] = np.ones(
                (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
            )
            tensors[f"{pfx}.self_attn.v_proj.weight"] = np.ones(
                (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
            )
        else:
            tensors[f"{pfx}.self_attn.k_proj.weight"] = np.zeros(
                (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
            )
            tensors[f"{pfx}.self_attn.v_proj.weight"] = np.zeros(
                (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
            )

        tensors[f"{pfx}.self_attn.o_proj.weight"] = np.zeros((_EXPECTED_HIDDEN_SIZE, 8), dtype=np.float32)
        tensors[f"{pfx}.mlp.gate_proj.weight"] = np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32)
        tensors[f"{pfx}.mlp.up_proj.weight"] = np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32)
        tensors[f"{pfx}.mlp.down_proj.weight"] = np.zeros((_EXPECTED_HIDDEN_SIZE, 16), dtype=np.float32)
        tensors[f"{pfx}.self_attn.q_norm.weight"] = np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32)
        tensors[f"{pfx}.self_attn.k_norm.weight"] = np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32)
        tensors[f"{pfx}.pre_feedforward_layernorm.weight"] = np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32)
        tensors[f"{pfx}.post_feedforward_layernorm.weight"] = np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32)

    metadata = {k: (_get_ptr(v), v.shape) for k, v in tensors.items()}
    llm = _core.init_model(metadata)

    assert llm["arch"] == "nano"
    expected_num_layers = 2
    assert llm["num_layers"] == expected_num_layers
    assert llm["kv_share_start"] == 1


def test_mojo_core_step_nano_kv_share_keeps_shared_layer_cache_slots_pristine() -> None:
    tensors = {
        "model.embed_tokens.weight": np.ones((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.norm.weight": np.ones((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "lm_head.weight": np.ones((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.per_layer_embed.weight": np.ones(
            (_EXPECTED_VOCAB_SIZE, 30, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.per_layer_embed.projection.weight": np.ones(
            (_EXPECTED_HIDDEN_SIZE, 30, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.per_layer_embed.norm.weight": np.ones((_EXPECTED_PER_LAYER_DIM_SMALL,), dtype=np.float32),
    }

    for layer_idx in range(2):
        pfx = f"model.layers.{layer_idx}"
        tensors[f"{pfx}.altup.router.weight"] = np.ones(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        )
        tensors[f"{pfx}.altup.router_norm.weight"] = np.ones((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32)
        tensors[f"{pfx}.altup.prediction_coefs"] = np.ones(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        )
        tensors[f"{pfx}.altup.correction_coefs"] = np.ones(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        )
        tensors[f"{pfx}.altup.output_scale"] = np.ones((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32)
        tensors[f"{pfx}.laurel.down_proj.weight"] = np.ones(
            (_EXPECTED_PER_LAYER_DIM_SMALL, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        )
        tensors[f"{pfx}.laurel.up_proj.weight"] = np.ones(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        )
        tensors[f"{pfx}.laurel.norm.weight"] = np.ones((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32)
        tensors[f"{pfx}.per_layer_map.gate.weight"] = np.ones(
            (_EXPECTED_PER_LAYER_DIM_SMALL, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        )
        tensors[f"{pfx}.per_layer_map.projection.weight"] = np.ones(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        )
        tensors[f"{pfx}.per_layer_map.norm.weight"] = np.ones((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32)
        tensors[f"{pfx}.input_layernorm.weight"] = np.ones((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32)
        tensors[f"{pfx}.post_attention_layernorm.weight"] = np.ones((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32)
        tensors[f"{pfx}.self_attn.q_proj.weight"] = np.ones((8, _EXPECTED_HIDDEN_SIZE), dtype=np.float32)

        if layer_idx == 0:
            tensors[f"{pfx}.self_attn.k_proj.weight"] = np.ones(
                (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
            )
            tensors[f"{pfx}.self_attn.v_proj.weight"] = np.ones(
                (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
            )
        else:
            tensors[f"{pfx}.self_attn.k_proj.weight"] = np.zeros(
                (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
            )
            tensors[f"{pfx}.self_attn.v_proj.weight"] = np.zeros(
                (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
            )

        tensors[f"{pfx}.self_attn.o_proj.weight"] = np.ones((_EXPECTED_HIDDEN_SIZE, 8), dtype=np.float32)
        tensors[f"{pfx}.mlp.gate_proj.weight"] = np.ones((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32)
        tensors[f"{pfx}.mlp.up_proj.weight"] = np.ones((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32)
        tensors[f"{pfx}.mlp.down_proj.weight"] = np.ones((_EXPECTED_HIDDEN_SIZE, 16), dtype=np.float32)
        tensors[f"{pfx}.self_attn.q_norm.weight"] = np.ones((_EXPECTED_HEAD_DIM,), dtype=np.float32)
        tensors[f"{pfx}.self_attn.k_norm.weight"] = np.ones((_EXPECTED_HEAD_DIM,), dtype=np.float32)
        tensors[f"{pfx}.pre_feedforward_layernorm.weight"] = np.ones((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32)
        tensors[f"{pfx}.post_feedforward_layernorm.weight"] = np.ones((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32)

    for i in range(3):
        tensors[f"model.altup.projection.{i}.weight"] = np.ones(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        )
        tensors[f"model.altup.unembed.{i}.weight"] = np.ones(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        )

    metadata = {k: (_get_ptr(v), v.shape) for k, v in tensors.items()}
    llm = _core.init_model(metadata)
    assert llm["arch"] == "nano"
    expected_num_layers = 2
    assert llm["num_layers"] == expected_num_layers
    assert llm["kv_share_start"] == 1

    _ = _core.step(llm, 1, 0.0, 0, 0.0)

    layer_span = llm["max_seq_len"] * llm["num_kv_heads"] * llm["head_dim"]
    k_cache = np.asarray(llm["k_cache"], dtype=np.float32)
    v_cache = np.asarray(llm["v_cache"], dtype=np.float32)

    assert np.count_nonzero(k_cache[:layer_span]) > 0
    assert np.count_nonzero(v_cache[:layer_span]) > 0
    assert np.count_nonzero(k_cache[layer_span : layer_span * 2]) == 0
    assert np.count_nonzero(v_cache[layer_span : layer_span * 2]) == 0


def test_mojo_core_embeddings_nano() -> None:
    tensors = {
        "model.embed_tokens.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "lm_head.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.per_layer_embed.weight": np.zeros(
            (_EXPECTED_VOCAB_SIZE, 30, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.per_layer_embed.projection.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, 30, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.per_layer_embed.norm.weight": np.zeros((_EXPECTED_PER_LAYER_DIM_SMALL,), dtype=np.float32),
        "model.layers.0.altup.router.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.altup.router_norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.altup.prediction_coefs": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.altup.correction_coefs": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.altup.output_scale": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.laurel.down_proj.weight": np.zeros(
            (_EXPECTED_PER_LAYER_DIM_SMALL, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.laurel.up_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.layers.0.laurel.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.per_layer_map.gate.weight": np.zeros(
            (_EXPECTED_PER_LAYER_DIM_SMALL, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.per_layer_map.projection.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.layers.0.per_layer_map.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.zeros((8, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.v_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.o_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 8), dtype=np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.up_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.down_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 16), dtype=np.float32),
        "model.layers.0.self_attn.q_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.self_attn.k_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.pre_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.altup.projection.0.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.altup.projection.1.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.altup.projection.2.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.altup.unembed.0.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.altup.unembed.1.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.altup.unembed.2.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
    }
    metadata = {k: (_get_ptr(v), v.shape) for k, v in tensors.items()}
    llm = _core.init_model(metadata)

    input_ids = np.array([[1, 2, 3]], dtype=np.int32)
    embeddings = _core.generate_embeddings(llm, input_ids)
    assert embeddings.shape == (1, _EXPECTED_HIDDEN_SIZE)


def test_mojo_core_standard_descriptor_cache_reused_across_calls() -> None:
    tensors = {
        "model.embed_tokens.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "lm_head.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.zeros((8, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.v_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.o_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 8), dtype=np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.up_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.down_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 16), dtype=np.float32),
        "model.layers.0.self_attn.q_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.self_attn.k_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.pre_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
    }
    metadata = {k: (_get_ptr(v), v.shape) for k, v in tensors.items()}
    llm = _core.init_model(metadata)
    if "descriptor_build_count" not in llm:
        pytest.skip("descriptor_build_count is unavailable in current compiled mogemma._core")
    assert llm["descriptor_build_count"] == 1

    _ = _core.step(llm, 1, 0.0, 0, 0.0)
    _ = _core.step(llm, 2, 0.0, 0, 0.0)
    embeddings = _core.generate_embeddings(llm, np.array([[1, 2, 3]], dtype=np.int32))

    assert embeddings.shape == (1, _EXPECTED_HIDDEN_SIZE)
    assert llm.get("descriptor_build_count", 1) == 1


def test_mojo_core_nano_descriptor_cache_reused_across_calls() -> None:
    tensors = {
        "model.embed_tokens.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "lm_head.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.per_layer_embed.weight": np.zeros(
            (_EXPECTED_VOCAB_SIZE, 30, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.per_layer_embed.projection.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, 30, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.per_layer_embed.norm.weight": np.zeros((_EXPECTED_PER_LAYER_DIM_SMALL,), dtype=np.float32),
        "model.layers.0.altup.router.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.altup.router_norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.altup.prediction_coefs": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.altup.correction_coefs": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.altup.output_scale": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.laurel.down_proj.weight": np.zeros(
            (_EXPECTED_PER_LAYER_DIM_SMALL, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.laurel.up_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.layers.0.laurel.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.per_layer_map.gate.weight": np.zeros(
            (_EXPECTED_PER_LAYER_DIM_SMALL, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.per_layer_map.projection.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.layers.0.per_layer_map.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.zeros((8, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.v_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.o_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 8), dtype=np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.up_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.down_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 16), dtype=np.float32),
        "model.layers.0.self_attn.q_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.self_attn.k_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.pre_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.altup.projection.0.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.altup.projection.1.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.altup.projection.2.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.altup.unembed.0.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.altup.unembed.1.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.altup.unembed.2.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
    }
    metadata = {k: (_get_ptr(v), v.shape) for k, v in tensors.items()}
    llm = _core.init_model(metadata)

    if "descriptor_build_count" not in llm:
        pytest.skip("descriptor_build_count is unavailable in current compiled mogemma._core")
    assert llm["arch"] == "nano"
    assert llm["descriptor_build_count"] == 1

    _ = _core.step(llm, 1, 0.0, 0, 0.0)
    _ = _core.step(llm, 2, 0.0, 0, 0.0)
    embeddings = _core.generate_embeddings(llm, np.array([[1, 2, 3]], dtype=np.int32))

    assert embeddings.shape == (1, _EXPECTED_HIDDEN_SIZE)
    assert llm.get("descriptor_build_count", 1) == 1


def test_mojo_core_step_nano_reuses_prepared_model_views() -> None:
    tensors = {
        "model.embed_tokens.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "lm_head.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.per_layer_embed.weight": np.zeros(
            (_EXPECTED_VOCAB_SIZE, 30, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.per_layer_embed.projection.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, 30, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.per_layer_embed.norm.weight": np.zeros((_EXPECTED_PER_LAYER_DIM_SMALL,), dtype=np.float32),
        "model.layers.0.altup.router.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.altup.router_norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.altup.prediction_coefs": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.altup.correction_coefs": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.altup.output_scale": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.laurel.down_proj.weight": np.zeros(
            (_EXPECTED_PER_LAYER_DIM_SMALL, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.laurel.up_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.layers.0.laurel.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.per_layer_map.gate.weight": np.zeros(
            (_EXPECTED_PER_LAYER_DIM_SMALL, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.per_layer_map.projection.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.layers.0.per_layer_map.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.zeros((8, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.v_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.o_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 8), dtype=np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.up_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.down_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 16), dtype=np.float32),
        "model.layers.0.self_attn.q_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.self_attn.k_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.pre_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.altup.projection.0.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.altup.projection.1.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.altup.projection.2.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.altup.unembed.0.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.altup.unembed.1.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.altup.unembed.2.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
    }
    metadata = {k: (_get_ptr(v), v.shape) for k, v in tensors.items()}
    llm = _core.init_model(metadata)

    assert llm["arch"] == "nano"
    assert llm["nano_model_build_count"] == 1

    _ = _core.step(llm, 1, 0.0, 0, 0.0)
    _ = _core.step(llm, 2, 0.0, 0, 0.0)
    expected_pos = 2
    assert llm["pos"] == expected_pos
    assert llm["nano_model_build_count"] == 1

    embeddings = _core.generate_embeddings(llm, np.array([[1, 2, 3]], dtype=np.int32))
    assert embeddings.shape == (1, _EXPECTED_HIDDEN_SIZE)
    expected_pos = 2
    assert llm["pos"] == expected_pos
    assert llm["nano_model_build_count"] == 1

    _ = _core.step(llm, 3, 0.0, 0, 0.0)
    expected_pos_3 = 3
    assert llm["pos"] == expected_pos_3


def test_mojo_core_step_standard_cpu_gpu_parity() -> None:
    if not hasattr(_core, "init_model_with_options"):
        pytest.skip("init_model_with_options is unavailable")

    tensors = {
        "model.embed_tokens.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "lm_head.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.zeros((8, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.v_proj.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        ),
        "model.layers.0.self_attn.o_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 8), dtype=np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.up_proj.weight": np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.layers.0.mlp.down_proj.weight": np.zeros((_EXPECTED_HIDDEN_SIZE, 16), dtype=np.float32),
        "model.layers.0.self_attn.q_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.self_attn.k_norm.weight": np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32),
        "model.layers.0.pre_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "model.layers.0.post_feedforward_layernorm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
    }

    # Populate with some non-zero data to make the math meaningful
    for k in tensors:
        tensors[k] += 0.1

    metadata = {k: (_get_ptr(v), v.shape) for k, v in tensors.items()}

    llm_cpu = _core.init_model_with_options(
        metadata,
        {},
        {"backend": "cpu", "device_kind": "cpu", "requested": "cpu", "strict": False, "availability_source": "auto"},
    )

    llm_gpu = _core.init_model_with_options(
        metadata,
        {},
        {"backend": "cuda", "device_kind": "gpu", "requested": "cuda", "strict": False, "availability_source": "auto"},
    )

    logits_cpu_1 = _core.step(llm_cpu, 1, 0.0, 0, 0.0)
    logits_gpu_1 = _core.step(llm_gpu, 1, 0.0, 0, 0.0)

    np.testing.assert_allclose(logits_cpu_1, logits_gpu_1, atol=1e-6, rtol=1e-5)

    logits_cpu_2 = _core.step(llm_cpu, 2, 0.0, 0, 0.0)
    logits_gpu_2 = _core.step(llm_gpu, 2, 0.0, 0, 0.0)

    np.testing.assert_allclose(logits_cpu_2, logits_gpu_2, atol=1e-6, rtol=1e-5)


def test_mojo_core_step_nano_cpu_gpu_parity() -> None:
    if not hasattr(_core, "init_model_with_options"):
        pytest.skip("init_model_with_options is unavailable")

    tensors = {
        "model.embed_tokens.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.norm.weight": np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32),
        "lm_head.weight": np.zeros((_EXPECTED_VOCAB_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32),
        "model.per_layer_embed.weight": np.zeros(
            (_EXPECTED_VOCAB_SIZE, 30, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.per_layer_embed.projection.weight": np.zeros(
            (_EXPECTED_HIDDEN_SIZE, 30, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        ),
        "model.per_layer_embed.norm.weight": np.zeros((_EXPECTED_PER_LAYER_DIM_SMALL,), dtype=np.float32),
    }

    for layer_idx in range(2):
        pfx = f"model.layers.{layer_idx}"
        tensors[f"{pfx}.altup.router.weight"] = np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        )
        tensors[f"{pfx}.altup.router_norm.weight"] = np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32)
        tensors[f"{pfx}.altup.prediction_coefs"] = np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        )
        tensors[f"{pfx}.altup.correction_coefs"] = np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        )
        tensors[f"{pfx}.altup.output_scale"] = np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32)

        tensors[f"{pfx}.laurel.down_proj.weight"] = np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        )
        tensors[f"{pfx}.laurel.up_proj.weight"] = np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        )
        tensors[f"{pfx}.laurel.norm.weight"] = np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32)

        tensors[f"{pfx}.per_layer_map.gate.weight"] = np.zeros(
            (_EXPECTED_PER_LAYER_DIM_SMALL, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        )
        tensors[f"{pfx}.per_layer_map.projection.weight"] = np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_PER_LAYER_DIM_SMALL), dtype=np.float32
        )
        tensors[f"{pfx}.per_layer_map.norm.weight"] = np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32)
        tensors[f"{pfx}.input_layernorm.weight"] = np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32)
        tensors[f"{pfx}.self_attn.q_proj.weight"] = np.zeros((8, _EXPECTED_HIDDEN_SIZE), dtype=np.float32)
        tensors[f"{pfx}.self_attn.k_proj.weight"] = np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        )
        tensors[f"{pfx}.self_attn.v_proj.weight"] = np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        )
        tensors[f"{pfx}.self_attn.o_proj.weight"] = np.zeros((_EXPECTED_HIDDEN_SIZE, 8), dtype=np.float32)
        tensors[f"{pfx}.mlp.gate_proj.weight"] = np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32)
        tensors[f"{pfx}.mlp.up_proj.weight"] = np.zeros((16, _EXPECTED_HIDDEN_SIZE), dtype=np.float32)
        tensors[f"{pfx}.mlp.down_proj.weight"] = np.zeros((_EXPECTED_HIDDEN_SIZE, 16), dtype=np.float32)
        tensors[f"{pfx}.self_attn.q_norm.weight"] = np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32)
        tensors[f"{pfx}.self_attn.k_norm.weight"] = np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32)
        tensors[f"{pfx}.self_attn.v_norm.weight"] = np.zeros((_EXPECTED_HEAD_DIM,), dtype=np.float32)
        tensors[f"{pfx}.pre_feedforward_layernorm.weight"] = np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32)
        tensors[f"{pfx}.post_feedforward_layernorm.weight"] = np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32)
        tensors[f"{pfx}.post_attention_layernorm.weight"] = np.zeros((_EXPECTED_HIDDEN_SIZE,), dtype=np.float32)

    for i in range(3):
        tensors[f"model.altup.projection.{i}.weight"] = np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        )
        tensors[f"model.altup.unembed.{i}.weight"] = np.zeros(
            (_EXPECTED_HIDDEN_SIZE, _EXPECTED_HIDDEN_SIZE), dtype=np.float32
        )

    # Populate with some non-zero data
    for k, v in tensors.items():
        tensors[k] = (v + 0.1).astype(np.float32)

    metadata = {k: (_get_ptr(v), v.shape) for k, v in tensors.items()}

    llm_cpu = _core.init_model_with_options(
        metadata,
        {},
        {"backend": "cpu", "device_kind": "cpu", "requested": "cpu", "strict": False, "availability_source": "auto"},
    )

    llm_gpu = _core.init_model_with_options(
        metadata,
        {},
        {"backend": "cuda", "device_kind": "gpu", "requested": "cuda", "strict": False, "availability_source": "auto"},
    )

    logits_cpu_1 = _core.step(llm_cpu, 1, 0.0, 0, 0.0)
    logits_gpu_1 = _core.step(llm_gpu, 1, 0.0, 0, 0.0)
    
    # Deterministic parity checkpoint to prevent Nano math regressions
    expected_logits = np.full(_EXPECTED_VOCAB_SIZE, 0.03999991, dtype=np.float32)
    np.testing.assert_allclose(logits_cpu_1, expected_logits, atol=1e-6)

    np.testing.assert_allclose(logits_cpu_1, logits_gpu_1, atol=1e-6, rtol=1e-5)

    logits_cpu_2 = _core.step(llm_cpu, 2, 0.0, 0, 0.0)
    logits_gpu_2 = _core.step(llm_gpu, 2, 0.0, 0, 0.0)

    np.testing.assert_allclose(logits_cpu_2, logits_gpu_2, atol=1e-6, rtol=1e-5)
