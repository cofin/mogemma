from testing import assert_true, assert_almost_equal
from memory import UnsafePointer
from collections import List

from mogemma.model import LaurelWeights, PerLayerMapWeights, AltUpWeights, TensorInfo
from mogemma.layers import forward_laurel, forward_per_layer_mapping, forward_altup, forward_nano_layer

fn alloc_zeros(size: Int) -> List[Float32]:
    return List[Float32](length=size, fill=0.0)

fn alloc_ones(size: Int) -> List[Float32]:
    return List[Float32](length=size, fill=1.0)

fn get_ptr(lst: List[Float32]) -> UnsafePointer[Float32, MutExternalOrigin]:
    return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(lst.unsafe_ptr()))

fn test_forward_laurel() raises:
    print("Testing forward_laurel...")
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
    var out_res = alloc_zeros(hidden_size)
    var out_norm = alloc_zeros(hidden_size)
    var scratch = alloc_zeros(bottleneck_dim * 2)
    
    forward_laurel(
        get_ptr(out_res),
        get_ptr(out_norm),
        get_ptr(x),
        get_ptr(x), # norm_res_ptr (using x as it's already all ones, normalized enough for test)
        weights,
        hidden_size,
        bottleneck_dim,
        get_ptr(scratch)
    )
    
    # x = [1, 1, 1, 1]
    # down_proj(x) = [4, 4]
    # up_proj([4, 4]) = [8, 8, 8, 8]
    # residual: [1, 1, 1, 1] + [8, 8, 8, 8] = [9, 9, 9, 9]
    # out_norm is norm([9, 9, 9, 9]) = [1, 1, 1, 1]
    
    for i in range(hidden_size):
        assert_almost_equal(out_res[i], 9.0, atol=1e-5)
        assert_almost_equal(out_norm[i], 1.0, atol=1e-5)
    
    _ = down_proj[0]
    _ = up_proj[0]
    _ = norm[0]
    _ = x[0]
    _ = out_res[0]
    _ = out_norm[0]
    _ = scratch[0]

fn test_forward_per_layer_mapping() raises:
    print("Testing forward_per_layer_mapping...")
    var hidden_size = 4
    var per_layer_dim = 2
    var layer_idx = 0
    var token_id = 0
    
    var weights = PerLayerMapWeights()
    var gate = alloc_ones(per_layer_dim * hidden_size)
    var projection = alloc_ones(hidden_size * per_layer_dim)
    var norm = alloc_ones(hidden_size)
    
    weights.gate = TensorInfo(Int(gate.unsafe_ptr()), per_layer_dim, hidden_size)
    weights.projection = TensorInfo(Int(projection.unsafe_ptr()), hidden_size, per_layer_dim)
    weights.norm = TensorInfo(Int(norm.unsafe_ptr()), hidden_size, 1)
    
    # per_layer_embed: [num_tokens, 30 layers, 256 dim]
    # For test: [1 token, 30 layers, 2 dim]
    var per_layer_embed = alloc_ones(1 * 30 * per_layer_dim)
    
    var x = alloc_ones(hidden_size)
    var out = alloc_zeros(hidden_size)
    var scratch = alloc_zeros(per_layer_dim * 10)
    
    forward_per_layer_mapping(
        get_ptr(out),
        get_ptr(x),
        get_ptr(x), # norm_x_ptr
        layer_idx,
        token_id,
        get_ptr(per_layer_embed),
        weights,
        hidden_size,
        per_layer_dim,
        get_ptr(scratch)
    )
    
    # x = [1, 1, 1, 1]
    # gate(x) = [4, 4]
    # embed lookup = [1, 1]
    # gated embed = [4, 4] * [1, 1] = [4, 4]
    # projection([4, 4]) = [8, 8, 8, 8]
    # residual: [1, 1, 1, 1] + [8, 8, 8, 8] = [9, 9, 9, 9]
    
    for i in range(hidden_size):
        assert_almost_equal(out[i], 9.0, atol=1e-5)

    _ = gate[0]
    _ = projection[0]
    _ = norm[0]
    _ = per_layer_embed[0]
    _ = x[0]
    _ = out[0]
    _ = scratch[0]

fn test_forward_altup() raises:
    print("Testing forward_altup...")
    var hidden_size = 4
    var num_variants = 4
    
    var weights = AltUpWeights()
    var router = alloc_ones(num_variants * hidden_size)
    var router_norm = alloc_ones(hidden_size)
    var prediction_coefs = alloc_ones(num_variants * hidden_size) # Actual shape in layers.mojo
    var correction_coefs = alloc_ones(hidden_size)
    var output_scale = alloc_ones(hidden_size)
    
    weights.router = TensorInfo(Int(router.unsafe_ptr()), num_variants, hidden_size)
    weights.router_norm = TensorInfo(Int(router_norm.unsafe_ptr()), hidden_size, 1)
    weights.prediction_coefs = TensorInfo(Int(prediction_coefs.unsafe_ptr()), num_variants, hidden_size)
    weights.correction_coefs = TensorInfo(Int(correction_coefs.unsafe_ptr()), hidden_size, 0)
    weights.output_scale = TensorInfo(Int(output_scale.unsafe_ptr()), hidden_size, 0)
    
    var projections = List[TensorInfo]()
    var projections_data = List[List[Float32]]()
    for i in range(num_variants - 1):
        var data = alloc_ones(hidden_size * hidden_size)
        projections_data.append(data.copy())
        projections.append(TensorInfo(Int(data.unsafe_ptr()), hidden_size, hidden_size))
        
    var unembeds = List[TensorInfo]()
    var unembeds_data = List[List[Float32]]()
    for i in range(num_variants - 1):
        var data = alloc_ones(hidden_size * hidden_size)
        unembeds_data.append(data.copy())
        unembeds.append(TensorInfo(Int(data.unsafe_ptr()), hidden_size, hidden_size))
        
    var x = alloc_ones(hidden_size)
    var out = alloc_zeros(hidden_size)
    var scratch = alloc_zeros(hidden_size * 40)
    
    forward_altup(
        get_ptr(out),
        get_ptr(x),
        get_ptr(x), # norm_res_ptr
        projections,
        unembeds,
        weights,
        hidden_size,
        num_variants,
        get_ptr(scratch)
    )
    
    # Just check it runs and outputs something non-zero for now
    var non_zero = False
    for i in range(hidden_size):
        if out[i] != 0:
            non_zero = True
            break
    assert_true(non_zero, "AltUp output should be non-zero")

    _ = router[0]
    _ = router_norm[0]
    _ = prediction_coefs[0]
    _ = correction_coefs[0]
    _ = output_scale[0]
    for i in range(num_variants - 1):
        _ = projections_data[i][0]
        _ = unembeds_data[i][0]
    _ = x[0]
    _ = out[0]
    _ = scratch[0]

from mogemma.model import LayerWeights, NanoLayerWeights

fn test_forward_nano_layer() raises:
    print("Testing forward_nano_layer...")
    var hidden_size = 4
    var intermediate_size = 4
    var num_heads = 2
    var num_kv_heads = 1
    var head_dim = 2
    var max_seq_len = 10
    var pos = 0
    var layer_idx = 0
    var token_id = 0
    var per_layer_dim = 2
    var bottleneck_dim = 2
    var num_variants = 4
    
    var weights = NanoLayerWeights()
    # Initialize base
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
    
    # Initialize altup
    var router = alloc_ones(num_variants * hidden_size)
    var router_norm = alloc_ones(hidden_size)
    var prediction_coefs = alloc_ones(num_variants * hidden_size)
    var correction_coefs = alloc_ones(hidden_size)
    var output_scale = alloc_ones(hidden_size)
    weights.altup.router = TensorInfo(Int(router.unsafe_ptr()), num_variants, hidden_size)
    weights.altup.router_norm = TensorInfo(Int(router_norm.unsafe_ptr()), hidden_size, 1)
    weights.altup.prediction_coefs = TensorInfo(Int(prediction_coefs.unsafe_ptr()), num_variants, hidden_size)
    weights.altup.correction_coefs = TensorInfo(Int(correction_coefs.unsafe_ptr()), hidden_size, 0)
    weights.altup.output_scale = TensorInfo(Int(output_scale.unsafe_ptr()), hidden_size, 0)
    
    # Initialize laurel
    var laurel_down = alloc_ones(bottleneck_dim * hidden_size)
    var laurel_up = alloc_ones(hidden_size * bottleneck_dim)
    var laurel_norm = alloc_ones(hidden_size)
    weights.laurel.down_proj = TensorInfo(Int(laurel_down.unsafe_ptr()), bottleneck_dim, hidden_size)
    weights.laurel.up_proj = TensorInfo(Int(laurel_up.unsafe_ptr()), hidden_size, bottleneck_dim)
    weights.laurel.norm = TensorInfo(Int(laurel_norm.unsafe_ptr()), hidden_size, 1)
    
    # Initialize per_layer_map
    var plm_gate = alloc_ones(per_layer_dim * hidden_size)
    var plm_proj = alloc_ones(hidden_size * per_layer_dim)
    var plm_norm = alloc_ones(hidden_size)
    weights.per_layer_map.gate = TensorInfo(Int(plm_gate.unsafe_ptr()), per_layer_dim, hidden_size)
    weights.per_layer_map.projection = TensorInfo(Int(plm_proj.unsafe_ptr()), hidden_size, per_layer_dim)
    weights.per_layer_map.norm = TensorInfo(Int(plm_norm.unsafe_ptr()), hidden_size, 1)
    
    var per_layer_embed = alloc_ones(1 * 30 * per_layer_dim)
    
    var altup_projections = List[TensorInfo]()
    var projections_data = List[List[Float32]]()
    for i in range(num_variants - 1):
        var data = alloc_ones(hidden_size * hidden_size)
        projections_data.append(data.copy())
        altup_projections.append(TensorInfo(Int(data.unsafe_ptr()), hidden_size, hidden_size))
        
    var altup_unembeds = List[TensorInfo]()
    var unembeds_data = List[List[Float32]]()
    for i in range(num_variants - 1):
        var data = alloc_ones(hidden_size * hidden_size)
        unembeds_data.append(data.copy())
        altup_unembeds.append(TensorInfo(Int(data.unsafe_ptr()), hidden_size, hidden_size))
        
    var x = alloc_ones(hidden_size)
    var out = alloc_zeros(hidden_size)
    var scratch = alloc_zeros(hidden_size * 200) # plenty
    
    var freqs_cos = alloc_zeros(head_dim)
    freqs_cos[0] = 1.0
    var freqs_sin = alloc_zeros(head_dim)
    freqs_sin[0] = 0.0
    
    var kv_cache_k = alloc_zeros(max_seq_len * num_kv_heads * head_dim)
    var kv_cache_v = alloc_zeros(max_seq_len * num_kv_heads * head_dim)
    
    forward_nano_layer(
        get_ptr(out),
        get_ptr(x),
        weights,
        layer_idx,
        token_id,
        pos,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        per_layer_dim,
        bottleneck_dim,
        get_ptr(freqs_cos),
        get_ptr(freqs_sin),
        get_ptr(kv_cache_k),
        get_ptr(kv_cache_v),
        max_seq_len,
        get_ptr(per_layer_embed),
        altup_projections,
        altup_unembeds,
        get_ptr(scratch)
    )
    
    var non_zero = False
    for i in range(hidden_size):
        if out[i] != 0:
            non_zero = True
            break
    assert_true(non_zero, "forward_nano_layer output should be non-zero")

    _ = input_layernorm[0]
    _ = post_attention_layernorm[0]
    _ = pre_ffw_layernorm[0]
    _ = post_ffw_layernorm[0]
    _ = q_proj[0]
    _ = k_proj[0]
    _ = v_proj[0]
    _ = o_proj[0]
    _ = gate_proj[0]
    _ = up_proj[0]
    _ = down_proj[0]
    _ = router[0]
    _ = router_norm[0]
    _ = prediction_coefs[0]
    _ = correction_coefs[0]
    _ = output_scale[0]
    _ = laurel_down[0]
    _ = laurel_up[0]
    _ = laurel_norm[0]
    _ = plm_gate[0]
    _ = plm_proj[0]
    _ = plm_norm[0]
    _ = per_layer_embed[0]
    _ = x[0]
    _ = out[0]
    _ = scratch[0]
    _ = freqs_cos[0]
    _ = freqs_sin[0]
    _ = kv_cache_k[0]
    _ = kv_cache_v[0]

fn main() raises:
    test_forward_laurel()
    test_forward_per_layer_mapping()
    test_forward_altup()
    test_forward_nano_layer()
    print("test_nano_layers.mojo passed!")
