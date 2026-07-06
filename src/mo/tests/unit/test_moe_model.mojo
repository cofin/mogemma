from mogemma.model import MoELayerWeights, TensorInfo


def test_moe_layer_weights_default_init() raises:
    var layer = MoELayerWeights()

    if layer.q_proj.shape_0 != 0 or layer.q_proj.shape_1 != 0:
        raise Error("q_proj should be zero-initialized")
    if layer.router_proj.shape_0 != 0 or layer.router_proj.shape_1 != 0:
        raise Error("router_proj should be zero-initialized")
    if layer.router_scale.shape_0 != 0 or layer.router_scale.shape_1 != 0:
        raise Error("router_scale should be zero-initialized")
    if layer.per_expert_scale.shape_0 != 0 or layer.per_expert_scale.shape_1 != 0:
        raise Error("per_expert_scale should be zero-initialized")
    if layer.dense_gate_proj.shape_0 != 0 or layer.dense_gate_proj.shape_1 != 0:
        raise Error("dense_gate_proj should be zero-initialized")
    if layer.dense_up_proj.shape_0 != 0 or layer.dense_up_proj.shape_1 != 0:
        raise Error("dense_up_proj should be zero-initialized")
    if layer.dense_down_proj.shape_0 != 0 or layer.dense_down_proj.shape_1 != 0:
        raise Error("dense_down_proj should be zero-initialized")
    if layer.expert_gate_up_proj.shape_0 != 0 or layer.expert_gate_up_proj.shape_1 != 0:
        raise Error("expert_gate_up_proj should be zero-initialized")
    if layer.expert_down_proj.shape_0 != 0 or layer.expert_down_proj.shape_1 != 0:
        raise Error("expert_down_proj should be zero-initialized")
    if layer.pre_feedforward_layernorm_2.shape_0 != 0 or layer.pre_feedforward_layernorm_2.shape_1 != 0:
        raise Error("pre_feedforward_layernorm_2 should be zero-initialized")
    if layer.post_feedforward_layernorm_1.shape_0 != 0 or layer.post_feedforward_layernorm_1.shape_1 != 0:
        raise Error("post_feedforward_layernorm_1 should be zero-initialized")
    if layer.post_feedforward_layernorm_2.shape_0 != 0 or layer.post_feedforward_layernorm_2.shape_1 != 0:
        raise Error("post_feedforward_layernorm_2 should be zero-initialized")
    if layer.post_feedforward_layernorm.shape_0 != 0 or layer.post_feedforward_layernorm.shape_1 != 0:
        raise Error("post_feedforward_layernorm should be zero-initialized")
    if layer.moe_skip_scale.shape_0 != 0 or layer.moe_skip_scale.shape_1 != 0:
        raise Error("moe_skip_scale should be zero-initialized")


def test_moe_layer_weights_new_layout_fields() raises:
    var layer = MoELayerWeights()
    layer.q_proj = TensorInfo(1, 64, 64)
    layer.k_proj = TensorInfo(2, 32, 64)
    layer.v_proj = TensorInfo(3, 32, 64)
    layer.o_proj = TensorInfo(4, 64, 64)
    layer.q_norm = TensorInfo(5, 16, 1)
    layer.k_norm = TensorInfo(6, 16, 1)
    layer.input_layernorm = TensorInfo(7, 64, 1)
    layer.post_attention_layernorm = TensorInfo(8, 64, 1)
    layer.pre_feedforward_layernorm = TensorInfo(9, 64, 1)
    layer.dense_gate_proj = TensorInfo(10, 128, 64)
    layer.dense_up_proj = TensorInfo(11, 128, 64)
    layer.dense_down_proj = TensorInfo(12, 64, 128)
    layer.post_feedforward_layernorm_1 = TensorInfo(13, 64, 1)
    layer.pre_feedforward_layernorm_2 = TensorInfo(14, 64, 1)
    layer.router_proj = TensorInfo(15, 8, 64)
    layer.router_scale = TensorInfo(16, 64, 1)
    layer.per_expert_scale = TensorInfo(17, 8, 1)
    layer.expert_gate_up_proj = TensorInfo(18, 8 * 256, 64)
    layer.expert_down_proj = TensorInfo(19, 8 * 64, 128)
    layer.post_feedforward_layernorm_2 = TensorInfo(20, 64, 1)
    layer.post_feedforward_layernorm = TensorInfo(21, 64, 1)
    layer.moe_skip_scale = TensorInfo(22, 1, 1)

    if layer.router_proj.shape_0 != 8 or layer.router_proj.shape_1 != 64:
        raise Error("router_proj shape mismatch")
    if layer.dense_gate_proj.shape_0 != 128:
        raise Error("dense_gate_proj shape mismatch")
    if layer.expert_gate_up_proj.shape_0 != 8 * 256:
        raise Error("expert_gate_up_proj shape mismatch")
    if layer.moe_skip_scale.shape_0 != 1:
        raise Error("moe_skip_scale shape mismatch")


def main() raises:
    test_moe_layer_weights_default_init()
    test_moe_layer_weights_new_layout_fields()
    print("test_moe_model.mojo passed!")
