from testing import assert_true, assert_almost_equal
from std.memory import UnsafePointer
from std.math import sqrt
from std.collections import List

from mogemma.model import (
    LayerWeights,
    ModelWeights,
    TensorInfo,
    KVCache,
    RoPETables,
    LAYER_TYPE_SLIDING,
    LAYER_TYPE_FULL,
)
from mogemma.layers import (
    forward_mlp,
    forward_sliding_attention,
    forward_full_attention,
    forward_gemma4_layer,
    forward_gemma4_step,
)
from mogemma.ops import CPUBackend


def alloc_zeros(size: Int) -> List[Float32]:
    return List[Float32](length=size, fill=0.0)


def alloc_ones(size: Int) -> List[Float32]:
    return List[Float32](length=size, fill=1.0)


def get_ptr(lst: List[Float32]) -> UnsafePointer[Float32, MutExternalOrigin]:
    return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(lst.unsafe_ptr()))


def test_forward_mlp() raises:
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
    var scratch = alloc_zeros(intermediate_size * 4)

    var backend = CPUBackend()
    forward_mlp(
        backend,
        get_ptr(out),
        get_ptr(x),
        weights,
        hidden_size,
        intermediate_size,
        get_ptr(scratch),
    )

    for i in range(hidden_size):
        assert_almost_equal(out[i], Float32(32.0), atol=2e-3)

    _ = gate_proj[0]
    _ = up_proj[0]
    _ = down_proj[0]
    _ = x[0]
    _ = out[0]
    _ = scratch[0]


def test_forward_sliding_attention() raises:
    var hidden_size = 4
    var num_heads = 2
    var num_kv_heads = 1
    var head_dim = 2
    var window_size = 8
    var max_context = 32

    var weights = LayerWeights()
    var q_proj = alloc_ones(num_heads * head_dim * hidden_size)
    var k_proj = alloc_ones(num_kv_heads * head_dim * hidden_size)
    var v_proj = alloc_ones(num_kv_heads * head_dim * hidden_size)
    var o_proj = alloc_ones(hidden_size * num_heads * head_dim)

    weights.q_proj = TensorInfo(Int(q_proj.unsafe_ptr()), num_heads * head_dim, hidden_size)
    weights.k_proj = TensorInfo(Int(k_proj.unsafe_ptr()), num_kv_heads * head_dim, hidden_size)
    weights.v_proj = TensorInfo(Int(v_proj.unsafe_ptr()), num_kv_heads * head_dim, hidden_size)
    weights.o_proj = TensorInfo(Int(o_proj.unsafe_ptr()), hidden_size, num_heads * head_dim)

    # Create a 1-layer KVCache (all sliding)
    var layer_types = List[UInt8](length=1, fill=UInt8(LAYER_TYPE_SLIDING))
    var lt_ptr = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(layer_types.unsafe_ptr()))
    var kv_cache = KVCache(1, num_kv_heads, head_dim, window_size, max_context, lt_ptr)

    # Create RoPETables
    var rope_tables = RoPETables(head_dim, 1.0, window_size, max_context)

    var x = alloc_ones(hidden_size)
    var out = alloc_zeros(hidden_size)
    var scratch = alloc_zeros(hidden_size * 20)

    var backend = CPUBackend()
    forward_sliding_attention(
        backend,
        get_ptr(out),
        get_ptr(x),
        weights,
        0,
        0,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_cache,
        rope_tables,
        False,
        0.0,
        get_ptr(scratch),
    )

    # Output should be non-zero
    var non_zero = False
    for i in range(hidden_size):
        if out[i] != 0.0:
            non_zero = True
    assert_true(non_zero, "forward_sliding_attention output should be non-zero")

    _ = q_proj[0]
    _ = k_proj[0]
    _ = v_proj[0]
    _ = o_proj[0]
    _ = x[0]
    _ = out[0]
    _ = scratch[0]
    _ = layer_types


def test_forward_full_attention() raises:
    var hidden_size = 4
    var num_heads = 2
    var num_kv_heads = 1
    var head_dim = 2
    var window_size = 8
    var max_context = 32

    var weights = LayerWeights()
    var q_proj = alloc_ones(num_heads * head_dim * hidden_size)
    var k_proj = alloc_ones(num_kv_heads * head_dim * hidden_size)
    var v_proj = alloc_ones(num_kv_heads * head_dim * hidden_size)
    var o_proj = alloc_ones(hidden_size * num_heads * head_dim)

    weights.q_proj = TensorInfo(Int(q_proj.unsafe_ptr()), num_heads * head_dim, hidden_size)
    weights.k_proj = TensorInfo(Int(k_proj.unsafe_ptr()), num_kv_heads * head_dim, hidden_size)
    weights.v_proj = TensorInfo(Int(v_proj.unsafe_ptr()), num_kv_heads * head_dim, hidden_size)
    weights.o_proj = TensorInfo(Int(o_proj.unsafe_ptr()), hidden_size, num_heads * head_dim)

    # Create a 1-layer KVCache (all full)
    var layer_types = List[UInt8](length=1, fill=UInt8(LAYER_TYPE_FULL))
    var lt_ptr = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(layer_types.unsafe_ptr()))
    var kv_cache = KVCache(1, num_kv_heads, head_dim, window_size, max_context, lt_ptr)

    # Create RoPETables with partial rotation
    var rope_tables = RoPETables(head_dim, 0.5, window_size, max_context)

    var x = alloc_ones(hidden_size)
    var out = alloc_zeros(hidden_size)
    var scratch = alloc_zeros(hidden_size * 20)

    var backend = CPUBackend()
    forward_full_attention(
        backend,
        get_ptr(out),
        get_ptr(x),
        weights,
        0,
        0,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        kv_cache,
        rope_tables,
        False,
        max_context,
        0.0,
        get_ptr(scratch),
    )

    # Output should be non-zero
    var non_zero = False
    for i in range(hidden_size):
        if out[i] != 0.0:
            non_zero = True
    assert_true(non_zero, "forward_full_attention output should be non-zero")

    _ = q_proj[0]
    _ = k_proj[0]
    _ = v_proj[0]
    _ = o_proj[0]
    _ = x[0]
    _ = out[0]
    _ = scratch[0]
    _ = layer_types


def main() raises:
    test_forward_mlp()
    test_forward_sliding_attention()
    test_forward_full_attention()
    print("test_layers.mojo passed!")
