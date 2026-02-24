from testing import assert_true, assert_almost_equal
from memory import UnsafePointer
from collections import List

from mogemma.model import LaurelWeights, PerLayerMapWeights, AltUpWeights, TensorInfo, NanoLayerWeights
from mogemma.layers import forward_laurel, forward_per_layer_mapping, forward_altup_predict, forward_altup_correct, forward_nano_layer

fn alloc_zeros(size: Int) -> List[Float32]:
    return List[Float32](length=size, fill=0.0)

fn alloc_ones(size: Int) -> List[Float32]:
    return List[Float32](length=size, fill=1.0)

fn get_ptr(lst: List[Float32]) -> UnsafePointer[Float32, MutExternalOrigin]:
    return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(lst.unsafe_ptr()))

fn test_forward_laurel() raises:
    var hidden_size = 4
    var bottleneck_dim = 2

    var weights = LaurelWeights()
    var down_proj = alloc_ones(bottleneck_dim * hidden_size)
    var up_proj = alloc_ones(hidden_size * bottleneck_dim)
    var norm = alloc_ones(hidden_size)

    weights.down_proj = TensorInfo(Int(down_proj.unsafe_ptr()), bottleneck_dim, hidden_size)
    weights.up_proj = TensorInfo(Int(up_proj.unsafe_ptr()), hidden_size, bottleneck_dim)
    weights.norm = TensorInfo(Int(norm.unsafe_ptr()), hidden_size, 1)

    var x = alloc_ones(hidden_size)
    var out = alloc_zeros(hidden_size)
    var scratch = alloc_zeros(bottleneck_dim + hidden_size * 2)

    forward_laurel(get_ptr(out), get_ptr(x), weights, hidden_size, bottleneck_dim, get_ptr(scratch))

    for i in range(hidden_size):
        assert_almost_equal(out[i], 2.0, atol=1e-5)

    _ = down_proj[0]
    _ = up_proj[0]
    _ = norm[0]
    _ = x[0]
    _ = out[0]
    _ = scratch[0]


fn test_forward_per_layer_mapping() raises:
    var hidden_size = 4
    var per_layer_dim = 2

    var weights = PerLayerMapWeights()
    var gate = alloc_ones(per_layer_dim * hidden_size)
    var projection = alloc_ones(hidden_size * per_layer_dim)
    var norm = alloc_ones(hidden_size)

    weights.gate = TensorInfo(Int(gate.unsafe_ptr()), per_layer_dim, hidden_size)
    weights.projection = TensorInfo(Int(projection.unsafe_ptr()), hidden_size, per_layer_dim)
    weights.norm = TensorInfo(Int(norm.unsafe_ptr()), hidden_size, 1)

    var active = alloc_ones(hidden_size)
    var per_layer_input = alloc_ones(per_layer_dim)
    var out = alloc_zeros(hidden_size)
    var scratch = alloc_zeros(hidden_size + per_layer_dim * 4)

    forward_per_layer_mapping(
        get_ptr(out),
        get_ptr(active),
        get_ptr(per_layer_input),
        weights,
        hidden_size,
        per_layer_dim,
        get_ptr(scratch),
    )

    # With all-ones weights and weighted RMSNorm, outputs settle near 1.0.
    for i in range(hidden_size):
        assert_almost_equal(out[i], 1.0, atol=1e-3)

    _ = gate[0]
    _ = projection[0]
    _ = norm[0]
    _ = active[0]
    _ = per_layer_input[0]
    _ = out[0]
    _ = scratch[0]


fn test_forward_altup_predict_and_correct() raises:
    var hidden_size = 4
    var num_modalities = 4

    var weights = AltUpWeights()
    var router = alloc_zeros(num_modalities * hidden_size)
    var router_norm = alloc_ones(hidden_size)
    var prediction_coefs = alloc_zeros(num_modalities * num_modalities * num_modalities)
    var correction_coefs = alloc_zeros(num_modalities * num_modalities)
    var output_scale = alloc_ones(hidden_size)

    weights.router = TensorInfo(Int(router.unsafe_ptr()), num_modalities, hidden_size)
    weights.router_norm = TensorInfo(Int(router_norm.unsafe_ptr()), hidden_size, 1)
    weights.prediction_coefs = TensorInfo(Int(prediction_coefs.unsafe_ptr()), num_modalities, num_modalities * num_modalities)
    weights.correction_coefs = TensorInfo(Int(correction_coefs.unsafe_ptr()), num_modalities, num_modalities)
    weights.output_scale = TensorInfo(Int(output_scale.unsafe_ptr()), hidden_size, 0)

    var streams = alloc_ones(num_modalities * hidden_size)
    var predictions = alloc_zeros(num_modalities * hidden_size)
    var corrected = alloc_zeros(num_modalities * hidden_size)
    var activated = alloc_ones(hidden_size)
    var scratch = alloc_zeros(hidden_size * 4)

    forward_altup_predict(
        get_ptr(predictions),
        get_ptr(streams),
        weights,
        hidden_size,
        num_modalities,
        get_ptr(scratch),
    )

    forward_altup_correct(
        get_ptr(corrected),
        get_ptr(predictions),
        get_ptr(activated),
        weights,
        hidden_size,
        num_modalities,
        get_ptr(scratch),
    )

    for i in range(num_modalities * hidden_size):
        assert_almost_equal(corrected[i], 1.0, atol=1e-5)

    _ = router[0]
    _ = router_norm[0]
    _ = prediction_coefs[0]
    _ = correction_coefs[0]
    _ = output_scale[0]
    _ = streams[0]
    _ = predictions[0]
    _ = corrected[0]
    _ = activated[0]
    _ = scratch[0]


fn test_forward_nano_layer() raises:
    var hidden_size = 4
    var intermediate_size = 4
    var num_heads = 2
    var num_kv_heads = 1
    var head_dim = 2
    var max_seq_len = 10
    var pos = 0
    var per_layer_dim = 2
    var num_modalities = 4

    var weights = NanoLayerWeights()

    var input_layernorm = alloc_ones(hidden_size)
    var post_attention_layernorm = alloc_ones(hidden_size)
    var pre_ffw_layernorm = alloc_ones(hidden_size)
    var post_ffw_layernorm = alloc_ones(hidden_size)
    var q_proj = alloc_ones(num_heads * head_dim * hidden_size)
    var k_proj = alloc_ones(num_kv_heads * head_dim * hidden_size)
    var v_proj = alloc_ones(num_kv_heads * head_dim * hidden_size)
    var o_proj = alloc_ones(hidden_size * num_heads * head_dim)
    var gate_proj = alloc_ones(intermediate_size * hidden_size)
    var up_proj = alloc_ones(intermediate_size * hidden_size)
    var down_proj = alloc_ones(hidden_size * intermediate_size)

    weights.base.input_layernorm = TensorInfo(Int(input_layernorm.unsafe_ptr()), hidden_size, 1)
    weights.base.post_attention_layernorm = TensorInfo(Int(post_attention_layernorm.unsafe_ptr()), hidden_size, 1)
    weights.base.pre_feedforward_layernorm = TensorInfo(Int(pre_ffw_layernorm.unsafe_ptr()), hidden_size, 1)
    weights.base.post_feedforward_layernorm = TensorInfo(Int(post_ffw_layernorm.unsafe_ptr()), hidden_size, 1)
    weights.base.q_proj = TensorInfo(Int(q_proj.unsafe_ptr()), num_heads * head_dim, hidden_size)
    weights.base.k_proj = TensorInfo(Int(k_proj.unsafe_ptr()), num_kv_heads * head_dim, hidden_size)
    weights.base.v_proj = TensorInfo(Int(v_proj.unsafe_ptr()), num_kv_heads * head_dim, hidden_size)
    weights.base.o_proj = TensorInfo(Int(o_proj.unsafe_ptr()), hidden_size, num_heads * head_dim)
    weights.base.gate_proj = TensorInfo(Int(gate_proj.unsafe_ptr()), intermediate_size, hidden_size)
    weights.base.up_proj = TensorInfo(Int(up_proj.unsafe_ptr()), intermediate_size, hidden_size)
    weights.base.down_proj = TensorInfo(Int(down_proj.unsafe_ptr()), hidden_size, intermediate_size)

    var router = alloc_ones(num_modalities * hidden_size)
    var router_norm = alloc_ones(hidden_size)
    var prediction_coefs = alloc_ones(num_modalities * num_modalities * num_modalities)
    var correction_coefs = alloc_ones(num_modalities * num_modalities)
    var output_scale = alloc_ones(hidden_size)
    weights.altup.router = TensorInfo(Int(router.unsafe_ptr()), num_modalities, hidden_size)
    weights.altup.router_norm = TensorInfo(Int(router_norm.unsafe_ptr()), hidden_size, 1)
    weights.altup.prediction_coefs = TensorInfo(Int(prediction_coefs.unsafe_ptr()), num_modalities, num_modalities * num_modalities)
    weights.altup.correction_coefs = TensorInfo(Int(correction_coefs.unsafe_ptr()), num_modalities, num_modalities)
    weights.altup.output_scale = TensorInfo(Int(output_scale.unsafe_ptr()), hidden_size, 0)

    var laurel_down = alloc_ones(per_layer_dim * hidden_size)
    var laurel_up = alloc_ones(hidden_size * per_layer_dim)
    var laurel_norm = alloc_ones(hidden_size)
    weights.laurel.down_proj = TensorInfo(Int(laurel_down.unsafe_ptr()), per_layer_dim, hidden_size)
    weights.laurel.up_proj = TensorInfo(Int(laurel_up.unsafe_ptr()), hidden_size, per_layer_dim)
    weights.laurel.norm = TensorInfo(Int(laurel_norm.unsafe_ptr()), hidden_size, 1)

    var plm_gate = alloc_ones(per_layer_dim * hidden_size)
    var plm_proj = alloc_ones(hidden_size * per_layer_dim)
    var plm_norm = alloc_ones(hidden_size)
    weights.per_layer_map.gate = TensorInfo(Int(plm_gate.unsafe_ptr()), per_layer_dim, hidden_size)
    weights.per_layer_map.projection = TensorInfo(Int(plm_proj.unsafe_ptr()), hidden_size, per_layer_dim)
    weights.per_layer_map.norm = TensorInfo(Int(plm_norm.unsafe_ptr()), hidden_size, 1)

    var in_streams = alloc_ones(num_modalities * hidden_size)
    var out_streams = alloc_zeros(num_modalities * hidden_size)
    var per_layer_input = alloc_ones(per_layer_dim)
    var scratch = alloc_zeros(hidden_size * 120)

    var freqs_cos = alloc_zeros(head_dim)
    freqs_cos[0] = 1.0
    var freqs_sin = alloc_zeros(head_dim)
    freqs_sin[0] = 0.0

    var kv_cache_k = alloc_zeros(max_seq_len * num_kv_heads * head_dim)
    var kv_cache_v = alloc_zeros(max_seq_len * num_kv_heads * head_dim)

    forward_nano_layer(
        get_ptr(out_streams),
        get_ptr(in_streams),
        weights,
        0,
        get_ptr(per_layer_input),
        pos,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        per_layer_dim,
        get_ptr(freqs_cos),
        get_ptr(freqs_sin),
        get_ptr(kv_cache_k),
        get_ptr(kv_cache_v),
        max_seq_len,
        num_modalities,
        True,
        get_ptr(scratch),
    )

    var non_zero = False
    for i in range(num_modalities * hidden_size):
        if out_streams[i] != 0:
            non_zero = True
            break
    assert_true(non_zero, "forward_nano_layer output should be non-zero")

    _ = out_streams[0]
    _ = scratch[0]


fn main() raises:
    test_forward_laurel()
    test_forward_per_layer_mapping()
    test_forward_altup_predict_and_correct()
    test_forward_nano_layer()
    print("test_nano_layers.mojo passed!")
