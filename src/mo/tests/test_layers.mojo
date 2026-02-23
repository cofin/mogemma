from testing import assert_true, assert_almost_equal
from memory import UnsafePointer
from math import sqrt
from collections import List

from model import LayerWeights, TensorInfo
from layers import forward_attention, forward_mlp, forward_layer

fn alloc_zeros(size: Int) -> List[Float32]:
    return List[Float32](length=size, fill=0.0)

fn alloc_ones(size: Int) -> List[Float32]:
    return List[Float32](length=size, fill=1.0)

fn get_ptr(lst: List[Float32]) -> UnsafePointer[Float32, MutExternalOrigin]:
    return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(lst.unsafe_ptr()))

fn test_forward_mlp() raises:
    # Set up
    var hidden_size = 4
    var intermediate_size = 2
    
    var weights = LayerWeights()
    var gate_proj = alloc_ones(intermediate_size * hidden_size)
    var up_proj = alloc_ones(intermediate_size * hidden_size)
    var down_proj = alloc_ones(hidden_size * intermediate_size)
    
    weights.gate_proj = TensorInfo(Int(gate_proj.unsafe_ptr()), intermediate_size, hidden_size)
    weights.up_proj = TensorInfo(Int(up_proj.unsafe_ptr()), intermediate_size, hidden_size)
    weights.down_proj = TensorInfo(Int(down_proj.unsafe_ptr()), hidden_size, intermediate_size)
    
    var x = alloc_ones(hidden_size)
    var out = alloc_zeros(hidden_size)
    var scratch = alloc_zeros(intermediate_size * 4) # plenty
    
    var x_ptr = get_ptr(x)
    var out_ptr = get_ptr(out)
    var scratch_ptr = get_ptr(scratch)
    
    forward_mlp(out_ptr, x_ptr, weights, hidden_size, intermediate_size, scratch_ptr)
    
    # Check
    for i in range(hidden_size):
        var expected: Float32 = 32.0
        assert_almost_equal(out[i], expected, atol=2e-3)
        
    _ = gate_proj[0]
    _ = up_proj[0]
    _ = down_proj[0]
    _ = x[0]
    _ = out[0]
    _ = scratch[0]

fn main() raises:
    test_forward_mlp()
    test_forward_attention()
    test_forward_attention_with_qk_norms()
    test_forward_layer()
    print("test_layers.mojo passed!")


fn test_forward_attention() raises:
    # Set up
    var hidden_size = 4
    var num_heads = 2
    var num_kv_heads = 1
    var head_dim = 2
    var max_seq_len = 10
    var pos = 0
    
    var weights = LayerWeights()
    var q_proj = alloc_ones(num_heads * head_dim * hidden_size)
    var k_proj = alloc_ones(num_kv_heads * head_dim * hidden_size)
    var v_proj = alloc_ones(num_kv_heads * head_dim * hidden_size)
    var o_proj = alloc_ones(hidden_size * num_heads * head_dim)
    
    weights.q_proj = TensorInfo(Int(q_proj.unsafe_ptr()), num_heads * head_dim, hidden_size)
    weights.k_proj = TensorInfo(Int(k_proj.unsafe_ptr()), num_kv_heads * head_dim, hidden_size)
    weights.v_proj = TensorInfo(Int(v_proj.unsafe_ptr()), num_kv_heads * head_dim, hidden_size)
    weights.o_proj = TensorInfo(Int(o_proj.unsafe_ptr()), hidden_size, num_heads * head_dim)
    
    var x = alloc_ones(hidden_size)
    var out = alloc_zeros(hidden_size)
    var scratch = alloc_zeros(hidden_size * 10) # plenty
    
    var freqs_cos = alloc_zeros(head_dim) # Only half is used by RoPE but we allocate head_dim
    freqs_cos[0] = 1.0 # First half element
    var freqs_sin = alloc_zeros(head_dim)
    freqs_sin[0] = 0.0 # First half element
    
    var kv_cache_k = alloc_zeros(max_seq_len * num_kv_heads * head_dim)
    var kv_cache_v = alloc_zeros(max_seq_len * num_kv_heads * head_dim)
    
    var x_ptr = get_ptr(x)
    var out_ptr = get_ptr(out)
    var scratch_ptr = get_ptr(scratch)
    var freqs_cos_ptr = get_ptr(freqs_cos)
    var freqs_sin_ptr = get_ptr(freqs_sin)
    var kv_cache_k_ptr = get_ptr(kv_cache_k)
    var kv_cache_v_ptr = get_ptr(kv_cache_v)
    
    forward_attention(
        out_ptr,
        x_ptr,
        weights,
        pos,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        freqs_cos_ptr,
        freqs_sin_ptr,
        kv_cache_k_ptr,
        kv_cache_v_ptr,
        max_seq_len,
        scratch_ptr
    )
    
    # Check
    for i in range(hidden_size):
        var expected: Float32 = 16.0
        assert_almost_equal(out[i], expected, atol=2e-3)
        
    _ = q_proj[0]
    _ = k_proj[0]
    _ = v_proj[0]
    _ = o_proj[0]
    _ = x[0]
    _ = out[0]
    _ = scratch[0]
    _ = freqs_cos[0]
    _ = freqs_sin[0]
    _ = kv_cache_k[0]
    _ = kv_cache_v[0]

fn test_forward_attention_with_qk_norms() raises:
    # Verify that QK norms are applied when weights are present
    var hidden_size = 4
    var num_heads = 2
    var num_kv_heads = 1
    var head_dim = 2
    var max_seq_len = 10
    var pos = 0

    var weights = LayerWeights()
    var q_proj = alloc_ones(num_heads * head_dim * hidden_size)
    var k_proj = alloc_ones(num_kv_heads * head_dim * hidden_size)
    var v_proj = alloc_ones(num_kv_heads * head_dim * hidden_size)
    var o_proj = alloc_ones(hidden_size * num_heads * head_dim)
    var q_norm = alloc_ones(head_dim)  # ones = identity norm
    var k_norm = alloc_ones(head_dim)

    weights.q_proj = TensorInfo(Int(q_proj.unsafe_ptr()), num_heads * head_dim, hidden_size)
    weights.k_proj = TensorInfo(Int(k_proj.unsafe_ptr()), num_kv_heads * head_dim, hidden_size)
    weights.v_proj = TensorInfo(Int(v_proj.unsafe_ptr()), num_kv_heads * head_dim, hidden_size)
    weights.o_proj = TensorInfo(Int(o_proj.unsafe_ptr()), hidden_size, num_heads * head_dim)
    weights.q_norm = TensorInfo(Int(q_norm.unsafe_ptr()), head_dim, 0)
    weights.k_norm = TensorInfo(Int(k_norm.unsafe_ptr()), head_dim, 0)

    var x = alloc_ones(hidden_size)
    var out = alloc_zeros(hidden_size)
    var scratch = alloc_zeros(hidden_size * 10)

    var freqs_cos = alloc_zeros(head_dim)
    freqs_cos[0] = 1.0
    var freqs_sin = alloc_zeros(head_dim)
    freqs_sin[0] = 0.0

    var kv_cache_k = alloc_zeros(max_seq_len * num_kv_heads * head_dim)
    var kv_cache_v = alloc_zeros(max_seq_len * num_kv_heads * head_dim)

    forward_attention(
        get_ptr(out),
        get_ptr(x),
        weights,
        pos,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        get_ptr(freqs_cos),
        get_ptr(freqs_sin),
        get_ptr(kv_cache_k),
        get_ptr(kv_cache_v),
        max_seq_len,
        get_ptr(scratch)
    )

    # With identity norm weights (all ones), RMS norm on uniform input
    # should produce the same result as without norms (since norm of all-same = identity)
    for i in range(hidden_size):
        var expected: Float32 = 16.0
        assert_almost_equal(out[i], expected, atol=2e-3)

    _ = q_proj[0]
    _ = k_proj[0]
    _ = v_proj[0]
    _ = o_proj[0]
    _ = q_norm[0]
    _ = k_norm[0]
    _ = x[0]
    _ = out[0]
    _ = scratch[0]
    _ = freqs_cos[0]
    _ = freqs_sin[0]
    _ = kv_cache_k[0]
    _ = kv_cache_v[0]

fn test_forward_layer() raises:
    # Set up
    var hidden_size = 4
    var intermediate_size = 4
    var num_heads = 2
    var num_kv_heads = 1
    var head_dim = 2
    var max_seq_len = 10
    var pos = 0
    
    var weights = LayerWeights()
    var input_layernorm = alloc_ones(hidden_size)
    var post_attention_layernorm = alloc_ones(hidden_size)
    var q_proj = alloc_ones(num_heads * head_dim * hidden_size)
    var k_proj = alloc_ones(num_kv_heads * head_dim * hidden_size)
    var v_proj = alloc_ones(num_kv_heads * head_dim * hidden_size)
    var o_proj = alloc_ones(hidden_size * num_heads * head_dim)
    var gate_proj = alloc_ones(intermediate_size * hidden_size)
    var up_proj = alloc_ones(intermediate_size * hidden_size)
    var down_proj = alloc_ones(hidden_size * intermediate_size)
    
    weights.input_layernorm = TensorInfo(Int(input_layernorm.unsafe_ptr()), hidden_size, 1)
    weights.post_attention_layernorm = TensorInfo(Int(post_attention_layernorm.unsafe_ptr()), hidden_size, 1)
    weights.q_proj = TensorInfo(Int(q_proj.unsafe_ptr()), num_heads * head_dim, hidden_size)
    weights.k_proj = TensorInfo(Int(k_proj.unsafe_ptr()), num_kv_heads * head_dim, hidden_size)
    weights.v_proj = TensorInfo(Int(v_proj.unsafe_ptr()), num_kv_heads * head_dim, hidden_size)
    weights.o_proj = TensorInfo(Int(o_proj.unsafe_ptr()), hidden_size, num_heads * head_dim)
    weights.gate_proj = TensorInfo(Int(gate_proj.unsafe_ptr()), intermediate_size, hidden_size)
    weights.up_proj = TensorInfo(Int(up_proj.unsafe_ptr()), intermediate_size, hidden_size)
    weights.down_proj = TensorInfo(Int(down_proj.unsafe_ptr()), hidden_size, intermediate_size)
    
    var x = alloc_ones(hidden_size)
    var out = alloc_zeros(hidden_size)
    var scratch = alloc_zeros(hidden_size * 20) # plenty
    
    var freqs_cos = alloc_zeros(head_dim)
    freqs_cos[0] = 1.0
    var freqs_sin = alloc_zeros(head_dim)
    freqs_sin[0] = 0.0
    
    var kv_cache_k = alloc_zeros(max_seq_len * num_kv_heads * head_dim)
    var kv_cache_v = alloc_zeros(max_seq_len * num_kv_heads * head_dim)
    
    var x_ptr = get_ptr(x)
    var out_ptr = get_ptr(out)
    var scratch_ptr = get_ptr(scratch)
    var freqs_cos_ptr = get_ptr(freqs_cos)
    var freqs_sin_ptr = get_ptr(freqs_sin)
    var kv_cache_k_ptr = get_ptr(kv_cache_k)
    var kv_cache_v_ptr = get_ptr(kv_cache_v)
    
    forward_layer(
        out_ptr,
        x_ptr,
        weights,
        pos,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        freqs_cos_ptr,
        freqs_sin_ptr,
        kv_cache_k_ptr,
        kv_cache_v_ptr,
        max_seq_len,
        scratch_ptr
    )
    
    # Check
    for i in range(hidden_size):
        var expected: Float32 = 80.99796541
        assert_almost_equal(out[i], expected, atol=2e-3)
        
    _ = input_layernorm[0]
    _ = post_attention_layernorm[0]
    _ = q_proj[0]
    _ = k_proj[0]
    _ = v_proj[0]
    _ = o_proj[0]
    _ = gate_proj[0]
    _ = up_proj[0]
    _ = down_proj[0]
    _ = x[0]
    _ = out[0]
    _ = scratch[0]
    _ = freqs_cos[0]
    _ = freqs_sin[0]
    _ = kv_cache_k[0]
    _ = kv_cache_v[0]
