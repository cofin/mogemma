from std.memory import UnsafePointer

from mogemma.core import _flatten_moe_weights, _hydrate_moe_weights
from mogemma.model import MoELayerWeights, MoEModelWeights, TensorInfo


def _make_test_layer(base: Int) -> MoELayerWeights:
    var layer = MoELayerWeights()
    layer.q_proj = TensorInfo(base + 1, 64, 64)
    layer.k_proj = TensorInfo(base + 2, 32, 64)
    layer.v_proj = TensorInfo(base + 3, 32, 64)
    layer.o_proj = TensorInfo(base + 4, 64, 64)
    layer.q_norm = TensorInfo(base + 5, 16, 1)
    layer.k_norm = TensorInfo(base + 6, 16, 1)
    layer.input_layernorm = TensorInfo(base + 7, 64, 1)
    layer.post_attention_layernorm = TensorInfo(base + 8, 64, 1)
    layer.pre_feedforward_layernorm = TensorInfo(base + 9, 64, 1)
    layer.dense_gate_proj = TensorInfo(base + 10, 128, 64)
    layer.dense_up_proj = TensorInfo(base + 11, 128, 64)
    layer.dense_down_proj = TensorInfo(base + 12, 64, 128)
    layer.post_feedforward_layernorm_1 = TensorInfo(base + 13, 64, 1)
    layer.pre_feedforward_layernorm_2 = TensorInfo(base + 14, 64, 1)
    layer.router_proj = TensorInfo(base + 15, 8, 64)
    layer.router_scale = TensorInfo(base + 16, 64, 1)
    layer.per_expert_scale = TensorInfo(base + 17, 8, 1)
    layer.expert_gate_up_proj = TensorInfo(base + 18, 8 * 256, 64)
    layer.expert_down_proj = TensorInfo(base + 19, 8 * 64, 128)
    layer.post_feedforward_layernorm_2 = TensorInfo(base + 20, 64, 1)
    layer.post_feedforward_layernorm = TensorInfo(base + 21, 64, 1)
    layer.moe_skip_scale = TensorInfo(base + 22, 1, 1)
    return layer


def test_moe_flatten_hydrate_round_trip() raises:
    var model = MoEModelWeights()
    model.embed_tokens = TensorInfo(100, 32000, 64)
    model.norm = TensorInfo(200, 64, 1)
    model.lm_head = TensorInfo(300, 32000, 64)
    model.layers.append(_make_test_layer(1000))

    var flat = _flatten_moe_weights(model)
    if len(flat) != 25 * 4:
        raise Error("Expected 25 TensorInfo records in flattened MoE model")

    var flat_ptr = UnsafePointer[Int, MutExternalOrigin](unsafe_from_address=Int(flat.unsafe_ptr()))
    var hydrated = _hydrate_moe_weights(flat_ptr, 1)

    if len(hydrated.layers) != 1:
        raise Error("Expected exactly one hydrated MoE layer")

    var layer = hydrated.layers[0]
    if Int(layer.router_proj.ptr) != 1015:
        raise Error("router_proj pointer mismatch after hydrate")
    if layer.dense_gate_proj.shape_0 != 128:
        raise Error("dense_gate_proj shape mismatch after hydrate")
    if layer.expert_gate_up_proj.shape_0 != 8 * 256:
        raise Error("expert_gate_up_proj shape mismatch after hydrate")
    if Int(layer.moe_skip_scale.ptr) != 1022:
        raise Error("moe_skip_scale pointer mismatch after hydrate")


def main() raises:
    test_moe_flatten_hydrate_round_trip()
    print("test_moe_core.mojo passed!")
