import numpy as np
import pytest
from mogemma import _core

def _get_ptr(arr):
    return arr.__array_interface__['data'][0]

def test_mojo_core_init_standard():
    # Allocate some real arrays
    tensors = {
        "model.embed_tokens.weight": np.zeros((10, 4), dtype=np.float32),
        "model.norm.weight": np.zeros((4,), dtype=np.float32),
        "lm_head.weight": np.zeros((10, 4), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.zeros((8, 4), dtype=np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.zeros((4, 4), dtype=np.float32),
        "model.layers.0.self_attn.v_proj.weight": np.zeros((4, 4), dtype=np.float32),
        "model.layers.0.self_attn.o_proj.weight": np.zeros((4, 8), dtype=np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.zeros((16, 4), dtype=np.float32),
        "model.layers.0.mlp.up_proj.weight": np.zeros((16, 4), dtype=np.float32),
        "model.layers.0.mlp.down_proj.weight": np.zeros((4, 16), dtype=np.float32),
        "model.layers.0.self_attn.q_norm.weight": np.zeros((2,), dtype=np.float32),
        "model.layers.0.self_attn.k_norm.weight": np.zeros((2,), dtype=np.float32),
        "model.layers.0.pre_feedforward_layernorm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.post_feedforward_layernorm.weight": np.zeros((4,), dtype=np.float32),
    }
    
    # Mock metadata
    metadata = {k: (_get_ptr(v), v.shape) for k, v in tensors.items()}
    
    llm = _core.init_model(metadata)
    assert llm["arch"] == "standard"
    assert llm["num_layers"] == 1
    assert llm["head_dim"] == 2
    assert llm["num_kv_heads"] == 2
    assert llm["hidden_size"] == 4
    assert llm["vocab_size"] == 10

def test_mojo_core_step_standard():
    tensors = {
        "model.embed_tokens.weight": np.zeros((10, 4), dtype=np.float32),
        "model.norm.weight": np.zeros((4,), dtype=np.float32),
        "lm_head.weight": np.zeros((10, 4), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.zeros((8, 4), dtype=np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.zeros((4, 4), dtype=np.float32),
        "model.layers.0.self_attn.v_proj.weight": np.zeros((4, 4), dtype=np.float32),
        "model.layers.0.self_attn.o_proj.weight": np.zeros((4, 8), dtype=np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.zeros((16, 4), dtype=np.float32),
        "model.layers.0.mlp.up_proj.weight": np.zeros((16, 4), dtype=np.float32),
        "model.layers.0.mlp.down_proj.weight": np.zeros((4, 16), dtype=np.float32),
        "model.layers.0.self_attn.q_norm.weight": np.zeros((2,), dtype=np.float32),
        "model.layers.0.self_attn.k_norm.weight": np.zeros((2,), dtype=np.float32),
        "model.layers.0.pre_feedforward_layernorm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.post_feedforward_layernorm.weight": np.zeros((4,), dtype=np.float32),
    }
    metadata = {k: (_get_ptr(v), v.shape) for k, v in tensors.items()}
    llm = _core.init_model(metadata)
    
    logits = _core.step(llm, 1, 0.0, 0, 0.0)
    assert logits.shape == (10,)
    assert llm["pos"] == 1

def test_mojo_core_init_nano():
    tensors = {
        "model.embed_tokens.weight": np.zeros((10, 4), dtype=np.float32),
        "model.norm.weight": np.zeros((4,), dtype=np.float32),
        "lm_head.weight": np.zeros((10, 4), dtype=np.float32),
        "model.per_layer_embed.weight": np.zeros((10, 30 * 2), dtype=np.float32),
        "model.per_layer_embed.projection.weight": np.zeros((4, 2), dtype=np.float32),
        "model.per_layer_embed.norm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.altup.router.weight": np.zeros((4, 4), dtype=np.float32),
        "model.layers.0.altup.router_norm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.altup.prediction_coefs": np.zeros((4, 4), dtype=np.float32),
        "model.layers.0.altup.correction_coefs": np.zeros((4,), dtype=np.float32),
        "model.layers.0.altup.output_scale": np.zeros((4,), dtype=np.float32),
        "model.layers.0.laurel.down_proj.weight": np.zeros((2, 4), dtype=np.float32),
        "model.layers.0.laurel.up_proj.weight": np.zeros((4, 2), dtype=np.float32),
        "model.layers.0.laurel.norm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.per_layer_map.gate.weight": np.zeros((2, 4), dtype=np.float32),
        "model.layers.0.per_layer_map.projection.weight": np.zeros((4, 2), dtype=np.float32),
        "model.layers.0.per_layer_map.norm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.zeros((8, 4), dtype=np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.zeros((4, 4), dtype=np.float32),
        "model.layers.0.self_attn.v_proj.weight": np.zeros((4, 4), dtype=np.float32),
        "model.layers.0.self_attn.o_proj.weight": np.zeros((4, 8), dtype=np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.zeros((16, 4), dtype=np.float32),
        "model.layers.0.mlp.up_proj.weight": np.zeros((16, 4), dtype=np.float32),
        "model.layers.0.mlp.down_proj.weight": np.zeros((4, 16), dtype=np.float32),
        "model.layers.0.self_attn.q_norm.weight": np.zeros((2,), dtype=np.float32),
        "model.layers.0.self_attn.k_norm.weight": np.zeros((2,), dtype=np.float32),
        "model.layers.0.pre_feedforward_layernorm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.post_feedforward_layernorm.weight": np.zeros((4,), dtype=np.float32),
    }
    
    metadata = {k: (_get_ptr(v), v.shape) for k, v in tensors.items()}
    
    llm = _core.init_model(metadata)
    assert llm["arch"] == "nano"
    assert llm["num_layers"] == 1
    assert llm["head_dim"] == 2
    assert llm["per_layer_dim"] == 2
    assert llm["bottleneck_dim"] == 2


def test_mojo_core_init_nano_with_3d_per_layer_embed():
    tensors = {
        "model.embed_tokens.weight": np.zeros((10, 4), dtype=np.float32),
        "model.norm.weight": np.zeros((4,), dtype=np.float32),
        "lm_head.weight": np.zeros((10, 4), dtype=np.float32),
        "model.per_layer_embed.weight": np.zeros((10, 30, 256), dtype=np.float32),
        "model.per_layer_embed.projection.weight": np.zeros((4, 30, 256), dtype=np.float32),
        "model.per_layer_embed.norm.weight": np.zeros((256,), dtype=np.float32),
        "model.layers.0.altup.router.weight": np.zeros((4, 4), dtype=np.float32),
        "model.layers.0.altup.router_norm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.altup.prediction_coefs": np.zeros((4, 4, 4), dtype=np.float32),
        "model.layers.0.altup.correction_coefs": np.zeros((4, 4), dtype=np.float32),
        "model.layers.0.altup.output_scale": np.zeros((4,), dtype=np.float32),
        "model.layers.0.laurel.down_proj.weight": np.zeros((2, 4), dtype=np.float32),
        "model.layers.0.laurel.up_proj.weight": np.zeros((4, 2), dtype=np.float32),
        "model.layers.0.laurel.norm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.per_layer_map.gate.weight": np.zeros((256, 4), dtype=np.float32),
        "model.layers.0.per_layer_map.projection.weight": np.zeros((4, 256), dtype=np.float32),
        "model.layers.0.per_layer_map.norm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.zeros((8, 4), dtype=np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.zeros((4, 4), dtype=np.float32),
        "model.layers.0.self_attn.v_proj.weight": np.zeros((4, 4), dtype=np.float32),
        "model.layers.0.self_attn.o_proj.weight": np.zeros((4, 8), dtype=np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.zeros((16, 4), dtype=np.float32),
        "model.layers.0.mlp.up_proj.weight": np.zeros((16, 4), dtype=np.float32),
        "model.layers.0.mlp.down_proj.weight": np.zeros((4, 16), dtype=np.float32),
        "model.layers.0.self_attn.q_norm.weight": np.zeros((2,), dtype=np.float32),
        "model.layers.0.self_attn.k_norm.weight": np.zeros((2,), dtype=np.float32),
        "model.layers.0.pre_feedforward_layernorm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.post_feedforward_layernorm.weight": np.zeros((4,), dtype=np.float32),
    }
    metadata = {k: (_get_ptr(v), v.shape) for k, v in tensors.items()}

    llm = _core.init_model(metadata)
    assert llm["arch"] == "nano"
    assert llm["per_layer_dim"] == 256

def test_mojo_core_step_nano():
    tensors = {
        "model.embed_tokens.weight": np.zeros((10, 4), dtype=np.float32),
        "model.norm.weight": np.zeros((4,), dtype=np.float32),
        "lm_head.weight": np.zeros((10, 4), dtype=np.float32),
        "model.per_layer_embed.weight": np.zeros((10, 30 * 2), dtype=np.float32),
        "model.per_layer_embed.projection.weight": np.zeros((4, 2), dtype=np.float32),
        "model.per_layer_embed.norm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.altup.router.weight": np.zeros((4, 4), dtype=np.float32),
        "model.layers.0.altup.router_norm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.altup.prediction_coefs": np.zeros((4, 4), dtype=np.float32),
        "model.layers.0.altup.correction_coefs": np.zeros((4,), dtype=np.float32),
        "model.layers.0.altup.output_scale": np.zeros((4,), dtype=np.float32),
        "model.layers.0.laurel.down_proj.weight": np.zeros((2, 4), dtype=np.float32),
        "model.layers.0.laurel.up_proj.weight": np.zeros((4, 2), dtype=np.float32),
        "model.layers.0.laurel.norm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.per_layer_map.gate.weight": np.zeros((2, 4), dtype=np.float32),
        "model.layers.0.per_layer_map.projection.weight": np.zeros((4, 2), dtype=np.float32),
        "model.layers.0.per_layer_map.norm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.zeros((8, 4), dtype=np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.zeros((4, 4), dtype=np.float32),
        "model.layers.0.self_attn.v_proj.weight": np.zeros((4, 4), dtype=np.float32),
        "model.layers.0.self_attn.o_proj.weight": np.zeros((4, 8), dtype=np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.zeros((16, 4), dtype=np.float32),
        "model.layers.0.mlp.up_proj.weight": np.zeros((16, 4), dtype=np.float32),
        "model.layers.0.mlp.down_proj.weight": np.zeros((4, 16), dtype=np.float32),
        "model.layers.0.self_attn.q_norm.weight": np.zeros((2,), dtype=np.float32),
        "model.layers.0.self_attn.k_norm.weight": np.zeros((2,), dtype=np.float32),
        "model.layers.0.pre_feedforward_layernorm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.post_feedforward_layernorm.weight": np.zeros((4,), dtype=np.float32),
        "model.altup.projection.0.weight": np.zeros((4, 4), dtype=np.float32),
        "model.altup.projection.1.weight": np.zeros((4, 4), dtype=np.float32),
        "model.altup.projection.2.weight": np.zeros((4, 4), dtype=np.float32),
        "model.altup.unembed.0.weight": np.zeros((4, 4), dtype=np.float32),
        "model.altup.unembed.1.weight": np.zeros((4, 4), dtype=np.float32),
        "model.altup.unembed.2.weight": np.zeros((4, 4), dtype=np.float32),
    }
    metadata = {k: (_get_ptr(v), v.shape) for k, v in tensors.items()}
    llm = _core.init_model(metadata)
    
    logits = _core.step(llm, 1, 0.0, 0, 0.0)
    assert logits.shape == (10,)
    assert llm["pos"] == 1

def test_mojo_core_embeddings_nano():
    tensors = {
        "model.embed_tokens.weight": np.zeros((10, 4), dtype=np.float32),
        "model.norm.weight": np.zeros((4,), dtype=np.float32),
        "lm_head.weight": np.zeros((10, 4), dtype=np.float32),
        "model.per_layer_embed.weight": np.zeros((10, 30 * 2), dtype=np.float32),
        "model.per_layer_embed.projection.weight": np.zeros((4, 2), dtype=np.float32),
        "model.per_layer_embed.norm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.altup.router.weight": np.zeros((4, 4), dtype=np.float32),
        "model.layers.0.altup.router_norm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.altup.prediction_coefs": np.zeros((4, 4), dtype=np.float32),
        "model.layers.0.altup.correction_coefs": np.zeros((4,), dtype=np.float32),
        "model.layers.0.altup.output_scale": np.zeros((4,), dtype=np.float32),
        "model.layers.0.laurel.down_proj.weight": np.zeros((2, 4), dtype=np.float32),
        "model.layers.0.laurel.up_proj.weight": np.zeros((4, 2), dtype=np.float32),
        "model.layers.0.laurel.norm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.per_layer_map.gate.weight": np.zeros((2, 4), dtype=np.float32),
        "model.layers.0.per_layer_map.projection.weight": np.zeros((4, 2), dtype=np.float32),
        "model.layers.0.per_layer_map.norm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.zeros((8, 4), dtype=np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.zeros((4, 4), dtype=np.float32),
        "model.layers.0.self_attn.v_proj.weight": np.zeros((4, 4), dtype=np.float32),
        "model.layers.0.self_attn.o_proj.weight": np.zeros((4, 8), dtype=np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.zeros((16, 4), dtype=np.float32),
        "model.layers.0.mlp.up_proj.weight": np.zeros((16, 4), dtype=np.float32),
        "model.layers.0.mlp.down_proj.weight": np.zeros((4, 16), dtype=np.float32),
        "model.layers.0.self_attn.q_norm.weight": np.zeros((2,), dtype=np.float32),
        "model.layers.0.self_attn.k_norm.weight": np.zeros((2,), dtype=np.float32),
        "model.layers.0.pre_feedforward_layernorm.weight": np.zeros((4,), dtype=np.float32),
        "model.layers.0.post_feedforward_layernorm.weight": np.zeros((4,), dtype=np.float32),
        "model.altup.projection.0.weight": np.zeros((4, 4), dtype=np.float32),
        "model.altup.projection.1.weight": np.zeros((4, 4), dtype=np.float32),
        "model.altup.projection.2.weight": np.zeros((4, 4), dtype=np.float32),
        "model.altup.unembed.0.weight": np.zeros((4, 4), dtype=np.float32),
        "model.altup.unembed.1.weight": np.zeros((4, 4), dtype=np.float32),
        "model.altup.unembed.2.weight": np.zeros((4, 4), dtype=np.float32),
    }
    metadata = {k: (_get_ptr(v), v.shape) for k, v in tensors.items()}
    llm = _core.init_model(metadata)
    
    input_ids = np.array([[1, 2, 3]], dtype=np.int32)
    embeddings = _core.generate_embeddings(llm, input_ids)
    assert embeddings.shape == (1, 4)
