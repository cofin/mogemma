import numpy as np
import numpy.typing as npt

from mogemma import _core

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
    assert llm["num_kv_heads"] == _EXPECTED_HEAD_DIM
    assert llm["hidden_size"] == _EXPECTED_HIDDEN_SIZE
    assert llm["vocab_size"] == _EXPECTED_VOCAB_SIZE


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
