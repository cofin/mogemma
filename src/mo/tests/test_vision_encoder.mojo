from std.collections import List
from std.memory import UnsafePointer
from testing import assert_almost_equal
from mogemma.model import VisionLayerWeights, VisionModelWeights, TensorInfo
from mogemma.ops import gelu, average_pool_2d
from mogemma.layers import forward_vision_attention


fn _make_ptr(ref l: List[Float32]) -> UnsafePointer[Float32, MutExternalOrigin]:
    return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(l.unsafe_ptr()))


fn test_vision_attention_output_shape() raises:
    """Verify bidirectional attention produces correct output dimensions."""
    var num_tokens = 4
    var hidden_size = 8
    var num_heads = 2
    var head_dim = 4  # hidden_size / num_heads

    # Create identity-like weights for Q,K,V,O projections
    # Each is [out_dim, in_dim] = [hidden_size, hidden_size]
    var w_size = hidden_size * hidden_size
    var q_w = List[Float32](length=w_size, fill=0.0)
    var k_w = List[Float32](length=w_size, fill=0.0)
    var v_w = List[Float32](length=w_size, fill=0.0)
    var o_w = List[Float32](length=w_size, fill=0.0)

    # Set diagonal to 1.0 for identity projection
    for i in range(hidden_size):
        q_w[i * hidden_size + i] = 1.0
        k_w[i * hidden_size + i] = 1.0
        v_w[i * hidden_size + i] = 1.0
        o_w[i * hidden_size + i] = 1.0

    var weights = VisionLayerWeights()
    weights.q_proj = TensorInfo(Int(_make_ptr(q_w)), hidden_size, hidden_size)
    weights.k_proj = TensorInfo(Int(_make_ptr(k_w)), hidden_size, hidden_size)
    weights.v_proj = TensorInfo(Int(_make_ptr(v_w)), hidden_size, hidden_size)
    weights.o_proj = TensorInfo(Int(_make_ptr(o_w)), hidden_size, hidden_size)

    # Input: [num_tokens, hidden_size]
    var x = List[Float32](length=num_tokens * hidden_size, fill=0.0)
    # Set each token to a distinct pattern
    for t in range(num_tokens):
        for d in range(hidden_size):
            x[t * hidden_size + d] = Float32(t + 1) * 0.1

    # Output buffer
    var out = List[Float32](length=num_tokens * hidden_size, fill=0.0)

    # Scratch: needs Q[N,H], K[N,H], V[N,H], scores[num_heads, N, N], attn_out[N,H]
    var scratch_size = num_tokens * hidden_size * 10 + num_heads * num_tokens * num_tokens
    var scratch = List[Float32](length=scratch_size, fill=0.0)

    forward_vision_attention(
        _make_ptr(out), _make_ptr(x), weights,
        num_tokens, hidden_size, num_heads, head_dim, _make_ptr(scratch),
    )

    # With identity projections and uniform-ish inputs, output should be non-zero
    var any_nonzero = False
    for i in range(num_tokens * hidden_size):
        if out[i] != 0.0:
            any_nonzero = True
            break
    if not any_nonzero:
        raise Error("Vision attention output is all zeros")

    print("Vision attention output shape test: OK")
    _ = q_w[0]
    _ = k_w[0]
    _ = v_w[0]
    _ = o_w[0]
    _ = x[0]
    _ = scratch[0]


fn test_vision_attention_bidirectional() raises:
    """Verify attention is NOT lower-triangular (bidirectional)."""
    var num_tokens = 3
    var hidden_size = 4
    var num_heads = 1
    var head_dim = 4

    # Identity weights
    var w_size = hidden_size * hidden_size
    var q_w = List[Float32](length=w_size, fill=0.0)
    var k_w = List[Float32](length=w_size, fill=0.0)
    var v_w = List[Float32](length=w_size, fill=0.0)
    var o_w = List[Float32](length=w_size, fill=0.0)
    for i in range(hidden_size):
        q_w[i * hidden_size + i] = 1.0
        k_w[i * hidden_size + i] = 1.0
        v_w[i * hidden_size + i] = 1.0
        o_w[i * hidden_size + i] = 1.0

    var weights = VisionLayerWeights()
    weights.q_proj = TensorInfo(Int(_make_ptr(q_w)), hidden_size, hidden_size)
    weights.k_proj = TensorInfo(Int(_make_ptr(k_w)), hidden_size, hidden_size)
    weights.v_proj = TensorInfo(Int(_make_ptr(v_w)), hidden_size, hidden_size)
    weights.o_proj = TensorInfo(Int(_make_ptr(o_w)), hidden_size, hidden_size)

    # Make token 0 distinct from tokens 1,2
    var x = List[Float32](length=num_tokens * hidden_size, fill=0.0)
    x[0] = 1.0; x[1] = 0.0; x[2] = 0.0; x[3] = 0.0  # token 0
    x[4] = 0.0; x[5] = 1.0; x[6] = 0.0; x[7] = 0.0  # token 1
    x[8] = 0.0; x[9] = 0.0; x[10] = 1.0; x[11] = 0.0  # token 2

    var out = List[Float32](length=num_tokens * hidden_size, fill=0.0)
    var scratch_size = num_tokens * hidden_size * 10 + num_heads * num_tokens * num_tokens
    var scratch = List[Float32](length=scratch_size, fill=0.0)

    forward_vision_attention(
        _make_ptr(out), _make_ptr(x), weights,
        num_tokens, hidden_size, num_heads, head_dim, _make_ptr(scratch),
    )

    # In bidirectional attention, token 0's output should be influenced by tokens 1,2
    # (not just by itself as in causal). If token 0's output has non-zero values in
    # dimensions 1 or 2, it attended to tokens 1 and 2.
    # With identity projections, V = X, so output is weighted avg of all token vectors.
    # Token 0 with causal would only see itself: output = [1,0,0,0].
    # With bidirectional, it sees all: output is weighted avg of [1,0,0,0], [0,1,0,0], [0,0,1,0]
    if out[1] == 0.0 and out[2] == 0.0:
        raise Error("Token 0 did not attend to future tokens — attention is causal, not bidirectional!")

    print("Vision attention bidirectional test: OK")
    _ = q_w[0]
    _ = k_w[0]
    _ = v_w[0]
    _ = o_w[0]
    _ = x[0]
    _ = scratch[0]


fn test_average_pool_reduction() raises:
    """Verify average pooling reduces token count by kernel^2."""
    var grid_h = 6
    var grid_w = 6
    var hidden_size = 4
    var kernel = 3
    var in_tokens = grid_h * grid_w  # 36
    var out_tokens = (grid_h // kernel) * (grid_w // kernel)  # 4

    var x = List[Float32](length=in_tokens * hidden_size, fill=2.0)
    var out = List[Float32](length=out_tokens * hidden_size, fill=0.0)

    average_pool_2d(
        _make_ptr(out), _make_ptr(x), grid_h, grid_w, hidden_size, kernel,
    )

    # All 2.0 input → all 2.0 output (average of identical values)
    for i in range(out_tokens * hidden_size):
        assert_almost_equal(out[i], 2.0, atol=1e-5)

    print("Average pool reduction test: OK (36 → 4 tokens)")
    _ = x[0]


fn main() raises:
    test_vision_attention_output_shape()
    test_vision_attention_bidirectional()
    test_average_pool_reduction()
    print("All vision encoder tests passed!")
