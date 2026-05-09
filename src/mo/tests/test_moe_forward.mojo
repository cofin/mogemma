from std.collections import List
from std.math import erf, exp, sqrt
from std.memory import UnsafePointer

from mogemma.core import _run_step
from mogemma.layers import forward_moe_layer
from mogemma.model import (
    KVCache,
    LAYER_TYPE_SLIDING,
    LayerWeights,
    ModelWeights,
    MoELayerWeights,
    MoEModelWeights,
    RoPETables,
    TensorInfo,
)
from mogemma.ops import CPUBackend


def _ptr(
    ref values: List[Float32],
) -> UnsafePointer[Float32, MutExternalOrigin]:
    return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(values.unsafe_ptr()))


def _i64_ptr(
    ref values: List[Int64],
) -> UnsafePointer[Int64, MutExternalOrigin]:
    return UnsafePointer[Int64, MutExternalOrigin](unsafe_from_address=Int(values.unsafe_ptr()))


def _zeros(size: Int) -> List[Float32]:
    return List[Float32](length=size, fill=0.0)


def _rms_norm_reference(values: List[Float32], weights: List[Float32]) -> List[Float32]:
    var sum_sq: Float32 = 0.0
    for i in range(len(values)):
        var value = values[i]
        sum_sq += value * value
    var inv_rms = 1.0 / sqrt(sum_sq / Float32(len(values)) + 1e-6)
    var out = List[Float32](length=len(values), fill=0.0)
    for i in range(len(values)):
        out[i] = values[i] * inv_rms * (1.0 + weights[i])
    return out^


def _rms_noscale_reference(values: List[Float32]) -> List[Float32]:
    var sum_sq: Float32 = 0.0
    for i in range(len(values)):
        var value = values[i]
        sum_sq += value * value
    var inv_rms = 1.0 / sqrt(sum_sq / Float32(len(values)) + 1e-6)
    var out = List[Float32](length=len(values), fill=0.0)
    for i in range(len(values)):
        out[i] = values[i] * inv_rms
    return out^


def _dot_1x2(row0: Float32, row1: Float32, values: List[Float32]) -> Float32:
    return row0 * values[0] + row1 * values[1]


def _gelu(value: Float32) -> Float32:
    return 0.5 * value * (1.0 + erf(value / 1.4142135623730951))


def test_forward_moe_layer_matches_reference() raises:
    var x = List[Float32](length=2, fill=0.0)
    x[0] = 1.0
    x[1] = 2.0
    var out = _zeros(2)
    var scratch = _zeros(128)

    var zeros2 = _zeros(2)
    var zeros4 = _zeros(4)
    var dense_gate = List[Float32](length=2, fill=0.0)
    dense_gate[0] = 1.0
    var dense_up = List[Float32](length=2, fill=0.0)
    dense_up[1] = 1.0
    var dense_down = List[Float32](length=2, fill=0.0)
    dense_down[0] = 3.0
    dense_down[1] = 4.0
    var pre_ff2 = List[Float32](length=2, fill=0.0)
    pre_ff2[1] = 1.0
    var router_proj = List[Float32](length=4, fill=0.0)
    router_proj[0] = 3.0
    router_proj[3] = 1.0
    var router_scale = List[Float32](length=2, fill=1.0)
    var per_expert_scale = List[Float32](length=2, fill=1.0)
    var expert_gate_up = List[Float32](length=8, fill=0.0)
    expert_gate_up[0] = 1.0
    expert_gate_up[3] = 1.0
    expert_gate_up[5] = 1.0
    expert_gate_up[6] = 1.0
    var expert_down = List[Float32](length=4, fill=0.0)
    expert_down[0] = 1.0
    expert_down[1] = 2.0
    expert_down[2] = 3.0
    expert_down[3] = 4.0

    var weights = MoELayerWeights()
    weights.q_proj = TensorInfo(Int(zeros4.unsafe_ptr()), 2, 2)
    weights.k_proj = TensorInfo(Int(zeros4.unsafe_ptr()), 2, 2)
    weights.v_proj = TensorInfo(Int(zeros4.unsafe_ptr()), 2, 2)
    weights.o_proj = TensorInfo(Int(zeros4.unsafe_ptr()), 2, 2)
    weights.input_layernorm = TensorInfo(Int(zeros2.unsafe_ptr()), 2, 1)
    weights.post_attention_layernorm = TensorInfo(Int(zeros2.unsafe_ptr()), 2, 1)
    weights.pre_feedforward_layernorm = TensorInfo(Int(zeros2.unsafe_ptr()), 2, 1)
    weights.dense_gate_proj = TensorInfo(Int(dense_gate.unsafe_ptr()), 1, 2)
    weights.dense_up_proj = TensorInfo(Int(dense_up.unsafe_ptr()), 1, 2)
    weights.dense_down_proj = TensorInfo(Int(dense_down.unsafe_ptr()), 2, 1)
    weights.post_feedforward_layernorm_1 = TensorInfo(Int(zeros2.unsafe_ptr()), 2, 1)
    weights.pre_feedforward_layernorm_2 = TensorInfo(Int(pre_ff2.unsafe_ptr()), 2, 1)
    weights.router_proj = TensorInfo(Int(router_proj.unsafe_ptr()), 2, 2)
    weights.router_scale = TensorInfo(Int(router_scale.unsafe_ptr()), 2, 1)
    weights.per_expert_scale = TensorInfo(Int(per_expert_scale.unsafe_ptr()), 2, 1)
    weights.expert_gate_up_proj = TensorInfo(Int(expert_gate_up.unsafe_ptr()), 2, 4)
    weights.expert_down_proj = TensorInfo(Int(expert_down.unsafe_ptr()), 2, 2)
    weights.post_feedforward_layernorm_2 = TensorInfo(Int(zeros2.unsafe_ptr()), 2, 1)
    weights.post_feedforward_layernorm = TensorInfo(Int(zeros2.unsafe_ptr()), 2, 1)
    var layer_scalar_buf = List[Float32](length=1, fill=0.5)
    weights.moe_skip_scale = TensorInfo(Int(layer_scalar_buf.unsafe_ptr()), 1, 1)

    var layer_types = List[UInt8](length=1, fill=UInt8(LAYER_TYPE_SLIDING))
    var kv_cache = KVCache(
        1,
        1,
        2,
        4,
        4,
        UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(layer_types.unsafe_ptr())),
    )
    var rope_tables = RoPETables(2, 1.0, 4, 4)
    var backend = CPUBackend()
    var dummy_stage = 0
    var dummy_ctx = 0

    forward_moe_layer(
        backend,
        _ptr(out),
        _ptr(x),
        weights,
        0,
        0,
        2,
        1,
        1,
        2,
        2,
        1,
        1,
        kv_cache,
        rope_tables,
        4,
        0.0,
        _ptr(scratch),
        dummy_stage,
        dummy_ctx,
    )

    var dense_norm = _rms_norm_reference(x, zeros2)
    var dense_hidden = _gelu(_dot_1x2(1.0, 0.0, dense_norm)) * _dot_1x2(0.0, 1.0, dense_norm)
    var dense_raw = List[Float32](length=2, fill=0.0)
    dense_raw[0] = dense_hidden * 3.0
    dense_raw[1] = dense_hidden * 4.0
    var dense_post = _rms_norm_reference(dense_raw, zeros2)

    var x_r = _rms_norm_reference(x, pre_ff2)
    var router_input = _rms_noscale_reference(x)
    var inv_hidden: Float32 = 1.0 / sqrt(Float32(2.0))
    for i in range(2):
        router_input[i] *= router_scale[i] * inv_hidden

    var logits = List[Float32](length=2, fill=0.0)
    logits[0] = _dot_1x2(3.0, 0.0, router_input)
    logits[1] = _dot_1x2(0.0, 1.0, router_input)
    var max_logit = logits[0]
    if logits[1] > max_logit:
        max_logit = logits[1]
    var exp0 = exp(logits[0] - max_logit)
    var exp1 = exp(logits[1] - max_logit)
    var select_expert = 0
    if exp1 > exp0:
        select_expert = 1

    var gate_value: Float32
    var up_value: Float32
    var down0: Float32
    var down1: Float32
    if select_expert == 0:
        gate_value = _dot_1x2(1.0, 0.0, x_r)
        up_value = _dot_1x2(0.0, 1.0, x_r)
        down0 = 1.0
        down1 = 2.0
    else:
        gate_value = _dot_1x2(0.0, 1.0, x_r)
        up_value = _dot_1x2(1.0, 0.0, x_r)
        down0 = 3.0
        down1 = 4.0

    var expert_hidden = _gelu(gate_value) * up_value
    var moe_raw = List[Float32](length=2, fill=0.0)
    moe_raw[0] = expert_hidden * down0
    moe_raw[1] = expert_hidden * down1
    var moe_post = _rms_norm_reference(moe_raw, zeros2)

    # HF-pinned formula: out = layer_scalar * (residual + post_ffw_norm(h1 + h2))
    var combine_raw = List[Float32](length=2, fill=0.0)
    for i in range(2):
        combine_raw[i] = dense_post[i] + moe_post[i]
    var combine_normed = _rms_norm_reference(combine_raw, zeros2)
    var expected = List[Float32](length=2, fill=0.0)
    for i in range(2):
        expected[i] = (x[i] + combine_normed[i]) * layer_scalar_buf[0]

    for i in range(2):
        var diff = out[i] - expected[i]
        if diff < 0.0:
            diff = -diff
        if diff > 5e-4:
            raise Error("forward_moe_layer reference mismatch at index " + String(i))


def test_run_step_dispatches_to_moe_model() raises:
    var std_embed = List[Float32](length=4, fill=0.0)
    std_embed[0] = 9.0
    std_embed[1] = 1.0
    std_embed[2] = 8.0
    std_embed[3] = 2.0
    var std_norm = _zeros(2)
    var std_head = List[Float32](length=4, fill=0.0)
    std_head[0] = 2.0
    std_head[3] = 3.0

    var moe_embed = List[Float32](length=4, fill=0.0)
    moe_embed[0] = 1.0
    moe_embed[1] = 2.0
    moe_embed[2] = 5.0
    moe_embed[3] = 6.0
    var moe_norm = _zeros(2)
    var moe_head = List[Float32](length=4, fill=0.0)
    moe_head[0] = 1.0
    moe_head[3] = 1.0

    var model = ModelWeights()
    model.embed_tokens = TensorInfo(Int(std_embed.unsafe_ptr()), 2, 2)
    model.norm = TensorInfo(Int(std_norm.unsafe_ptr()), 2, 1)
    model.lm_head = TensorInfo(Int(std_head.unsafe_ptr()), 2, 2)
    model.layers = []

    var moe_model = MoEModelWeights()
    moe_model.embed_tokens = TensorInfo(Int(moe_embed.unsafe_ptr()), 2, 2)
    moe_model.norm = TensorInfo(Int(moe_norm.unsafe_ptr()), 2, 1)
    moe_model.lm_head = TensorInfo(Int(moe_head.unsafe_ptr()), 2, 2)

    var logits = _zeros(2)
    var scratch = _zeros(16)
    var layer_types = List[UInt8](length=1, fill=UInt8(LAYER_TYPE_SLIDING))
    var kv_cache = KVCache(
        0,
        1,
        2,
        4,
        4,
        UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(layer_types.unsafe_ptr())),
    )
    var rope_tables = RoPETables(2, 1.0, 4, 4)
    var kv_map = List[Int64](length=1, fill=0)
    var backend = CPUBackend()
    var dummy_stage = 0
    var dummy_ctx = 0
    var dummy_persistent = 0

    _run_step[CPUBackend, KVCache, Int, Int, Int](
        backend,
        _ptr(logits),
        0,
        0,
        model,
        2,
        1,
        1,
        2,
        1,
        2,
        kv_cache,
        rope_tables,
        False,
        4,
        _ptr(scratch),
        1,
        False,
        0,
        _i64_ptr(kv_map),
        0,
        moe_model,
        1,
        1,
        0.0,
        0.0,
        dummy_stage,
        dummy_ctx,
        dummy_persistent,
    )

    var scaled_embed = List[Float32](length=2, fill=0.0)
    var embed_scale: Float32 = sqrt(Float32(2.0))
    scaled_embed[0] = moe_embed[0] * embed_scale
    scaled_embed[1] = moe_embed[1] * embed_scale
    var expected = _rms_norm_reference(scaled_embed, moe_norm)

    for i in range(2):
        var diff = logits[i] - expected[i]
        if diff < 0.0:
            diff = -diff
        if diff > 5e-4:
            raise Error("_run_step should dispatch to the MoE model when num_experts > 0")


def main() raises:
    test_forward_moe_layer_matches_reference()
    test_run_step_dispatches_to_moe_model()
    print("test_moe_forward.mojo passed!")
