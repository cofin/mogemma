from std.collections import List
from mogemma.ops import (
    rms_norm,
    geglu,
    rope_rotate,
    vec_mat_mul,
    mat_mat_mul,
    softmax,
    softcap,
    gelu,
    average_pool_2d,
    top_k,
    CPUBackend,
    ComputeBackend,
)
from std.memory import UnsafePointer
from std.testing import assert_almost_equal


def test_rms_norm() raises:
    # 4 elements: all 1.0. Mean sq = 1.0, inv_rms ~= 1.0
    var x = List[Float32](length=4, fill=1.0)
    var w = List[Float32](length=4, fill=2.0)
    var out = List[Float32](length=4, fill=0.0)

    # We must cast the pointers
    var x_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(x.unsafe_ptr()))
    var w_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(w.unsafe_ptr()))
    var out_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(out.unsafe_ptr()))

    rms_norm[2](out_ptr, x_ptr, w_ptr, 4)

    # Runtime uses Gemma-style scale of (1 + weight)
    assert_almost_equal(out[0], 3.0, atol=1e-5)
    assert_almost_equal(out[1], 3.0, atol=1e-5)
    assert_almost_equal(out[2], 3.0, atol=1e-5)
    assert_almost_equal(out[3], 3.0, atol=1e-5)
    _ = x[0]
    _ = w[0]


def test_geglu() raises:
    # gate=1.0, up=2.0 -> gelu_gate = 0.5 * 1.0 * (1 + erf(1/sqrt(2))) ~= 0.5 * 1 * 1.84134 = 0.84134
    # out = 0.84134 * 2.0 = 1.68268
    var gate = List[Float32](length=4, fill=1.0)
    var up = List[Float32](length=4, fill=2.0)
    var out = List[Float32](length=4, fill=0.0)

    var gate_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(gate.unsafe_ptr()))
    var up_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(up.unsafe_ptr()))
    var out_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(out.unsafe_ptr()))

    geglu[2](out_ptr, gate_ptr, up_ptr, 4)

    assert_almost_equal(out[0], 1.68268, atol=1e-4)
    _ = gate[0]
    _ = up[0]


def test_rope_rotate() raises:
    var vec = List[Float32](length=4, fill=0.0)
    vec[0] = 1.0
    vec[1] = 0.0
    vec[2] = 0.0
    vec[3] = 1.0

    var cos = List[Float32](length=2, fill=0.0)
    cos[0] = 0.0
    cos[1] = 1.0

    var sin = List[Float32](length=2, fill=0.0)
    sin[0] = 1.0
    sin[1] = 0.0

    var vec_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(vec.unsafe_ptr()))
    var cos_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(cos.unsafe_ptr()))
    var sin_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(sin.unsafe_ptr()))

    rope_rotate[1](vec_ptr, cos_ptr, sin_ptr, 4)

    assert_almost_equal(vec[0], 0.0, atol=1e-5)
    assert_almost_equal(vec[2], 1.0, atol=1e-5)
    assert_almost_equal(vec[1], 0.0, atol=1e-5)
    assert_almost_equal(vec[3], 1.0, atol=1e-5)
    _ = cos[0]
    _ = sin[0]


def test_vec_mat_mul() raises:
    var x = List[Float32](length=4, fill=1.0)
    var w = List[Float32](length=8, fill=2.0)  # 2x4
    var out = List[Float32](length=2, fill=0.0)

    var x_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(x.unsafe_ptr()))
    var w_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(w.unsafe_ptr()))
    var out_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(out.unsafe_ptr()))

    vec_mat_mul[2](out_ptr, x_ptr, w_ptr, 4, 2)

    assert_almost_equal(out[0], 8.0, atol=1e-5)
    assert_almost_equal(out[1], 8.0, atol=1e-5)
    _ = x[0]
    _ = w[0]


def test_mat_mat_mul() raises:
    var batch_size = 2
    var in_dim = 4
    var out_dim = 2
    var x = List[Float32](length=batch_size * in_dim, fill=1.0)
    var w = List[Float32](length=out_dim * in_dim, fill=2.0)  # transposed [out_dim, in_dim]
    var out = List[Float32](length=batch_size * out_dim, fill=0.0)

    var x_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(x.unsafe_ptr()))
    var w_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(w.unsafe_ptr()))
    var out_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(out.unsafe_ptr()))

    mat_mat_mul[2](out_ptr, x_ptr, w_ptr, batch_size, in_dim, out_dim)

    # 1.0 * 2.0 * 4 = 8.0 for each output element
    assert_almost_equal(out[0], 8.0, atol=1e-5)
    assert_almost_equal(out[1], 8.0, atol=1e-5)
    assert_almost_equal(out[2], 8.0, atol=1e-5)
    assert_almost_equal(out[3], 8.0, atol=1e-5)
    _ = x[0]
    _ = w[0]


from mogemma.ops import vec_mat_mul_i8, mat_mat_mul_i8


def test_mat_mat_mul_i8() raises:
    var batch_size = 2
    var in_dim = 4
    var out_dim = 2
    var x = List[Float32](length=batch_size * in_dim, fill=1.0)
    var w = List[Int8](length=out_dim * in_dim, fill=10)  # Int8 weights, value=10
    var scale = List[Float32](length=1, fill=0.2)
    var out = List[Float32](length=batch_size * out_dim, fill=0.0)

    var x_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(x.unsafe_ptr()))
    var w_ptr = UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=Int(w.unsafe_ptr()))
    var scale_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(scale.unsafe_ptr()))
    var out_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(out.unsafe_ptr()))

    mat_mat_mul_i8[2](out_ptr, x_ptr, w_ptr, scale_ptr, batch_size, in_dim, out_dim)

    # Dequantized weight = 10 * 0.2 = 2.0
    # 1.0 * 2.0 * 4 = 8.0 for each output element
    assert_almost_equal(out[0], 8.0, atol=1e-5)
    assert_almost_equal(out[1], 8.0, atol=1e-5)
    assert_almost_equal(out[2], 8.0, atol=1e-5)
    assert_almost_equal(out[3], 8.0, atol=1e-5)
    _ = x[0]
    _ = w[0]
    _ = scale[0]


def test_softmax() raises:
    var x = List[Float32](length=3, fill=0.0)
    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0
    var x_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(x.unsafe_ptr()))
    softmax[1](x_ptr, 3)

    # expected: [0.09003057, 0.24472847, 0.66524096]
    assert_almost_equal(x[0], 0.09003057, atol=1e-5)
    assert_almost_equal(x[1], 0.24472847, atol=1e-5)
    assert_almost_equal(x[2], 0.66524096, atol=1e-5)
    _ = x[0]


def test_softcap() raises:
    var x = List[Float32](length=5, fill=0.0)
    x[0] = -100.0
    x[1] = -30.0
    x[2] = 0.0
    x[3] = 30.0
    x[4] = 100.0

    var x_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(x.unsafe_ptr()))
    softcap[1](x_ptr, 5, 30.0)

    assert_almost_equal(x[0], -29.9237, atol=1e-3)
    assert_almost_equal(x[1], -22.8478, atol=1e-3)
    assert_almost_equal(x[2], 0.0, atol=1e-6)
    assert_almost_equal(x[3], 22.8478, atol=1e-3)
    assert_almost_equal(x[4], 29.9237, atol=1e-3)


def test_softcap_zero_cap_is_noop() raises:
    var x = List[Float32](length=2, fill=0.0)
    x[0] = -100.0
    x[1] = 100.0

    var x_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(x.unsafe_ptr()))
    softcap[1](x_ptr, 2, 0.0)

    assert_almost_equal(x[0], -100.0, atol=1e-6)
    assert_almost_equal(x[1], 100.0, atol=1e-6)


def test_softcap_default_width_with_tail() raises:
    var size = 17
    var x = List[Float32](length=size, fill=1.0)
    x[0] = -100.0
    x[15] = 30.0
    x[16] = 100.0

    var x_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(x.unsafe_ptr()))
    softcap(x_ptr, size, 30.0)

    assert_almost_equal(x[0], -29.9237, atol=1e-3)
    assert_almost_equal(x[15], 22.8478, atol=1e-3)
    assert_almost_equal(x[16], 29.9237, atol=1e-3)


def test_gelu() raises:
    # gelu(1.0) = 0.5 * 1.0 * (1 + erf(1/sqrt(2))) ≈ 0.8413
    var x = List[Float32](length=4, fill=1.0)
    var out = List[Float32](length=4, fill=0.0)

    var x_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(x.unsafe_ptr()))
    var out_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(out.unsafe_ptr()))

    gelu[2](out_ptr, x_ptr, 4)

    assert_almost_equal(out[0], 0.8413, atol=1e-3)
    assert_almost_equal(out[1], 0.8413, atol=1e-3)

    # gelu(0.0) = 0.0
    var zero = List[Float32](length=2, fill=0.0)
    var zero_out = List[Float32](length=2, fill=99.0)
    var zero_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(zero.unsafe_ptr()))
    var zero_out_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(zero_out.unsafe_ptr()))
    gelu[1](zero_out_ptr, zero_ptr, 2)
    assert_almost_equal(zero_out[0], 0.0, atol=1e-6)

    # gelu(-1.0) ≈ -0.1587
    var neg = List[Float32](length=2, fill=-1.0)
    var neg_out = List[Float32](length=2, fill=0.0)
    var neg_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(neg.unsafe_ptr()))
    var neg_out_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(neg_out.unsafe_ptr()))
    gelu[1](neg_out_ptr, neg_ptr, 2)
    assert_almost_equal(neg_out[0], -0.1587, atol=1e-3)

    _ = x[0]
    _ = zero[0]
    _ = neg[0]


def test_average_pool_2d() raises:
    # 6x6 grid, hidden_size=2, kernel=3
    # Output should be 2x2 grid, hidden_size=2
    var grid_h = 6
    var grid_w = 6
    var hidden_size = 2
    var kernel = 3
    var in_tokens = grid_h * grid_w  # 36
    var out_h = grid_h // kernel  # 2
    var out_w = grid_w // kernel  # 2
    var out_tokens = out_h * out_w  # 4

    var x = List[Float32](length=in_tokens * hidden_size, fill=1.0)
    var out = List[Float32](length=out_tokens * hidden_size, fill=0.0)

    var x_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(x.unsafe_ptr()))
    var out_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(out.unsafe_ptr()))

    average_pool_2d(out_ptr, x_ptr, grid_h, grid_w, hidden_size, kernel)

    # All inputs are 1.0, so all averages should be 1.0
    for i in range(out_tokens * hidden_size):
        assert_almost_equal(out[i], 1.0, atol=1e-5)

    # Test with varying values: set first 3x3 block's hidden[0] to sequential 1-9
    for i in range(in_tokens * hidden_size):
        x[i] = 0.0

    # First 3x3 block (rows 0-2, cols 0-2) = values 1..9 in hidden dim 0
    var val: Float32 = 1.0
    for r in range(3):
        for c in range(3):
            var idx = (r * grid_w + c) * hidden_size
            x[idx] = val
            val += 1.0

    average_pool_2d(out_ptr, x_ptr, grid_h, grid_w, hidden_size, kernel)

    # Average of 1..9 = 45/9 = 5.0
    assert_almost_equal(out[0], 5.0, atol=1e-5)

    _ = x[0]


def test_top_k() raises:
    # Values: [0.1, 0.5, 0.3, 0.9, 0.2, 0.8, 0.4, 0.7]
    var vals = List[Float32](length=8, fill=0.0)
    vals[0] = 0.1
    vals[1] = 0.5
    vals[2] = 0.3
    vals[3] = 0.9
    vals[4] = 0.2
    vals[5] = 0.8
    vals[6] = 0.4
    vals[7] = 0.7

    var out_idx = List[Int32](length=3, fill=0)
    var out_vals = List[Float32](length=3, fill=0.0)

    var vals_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(vals.unsafe_ptr()))
    var idx_ptr = UnsafePointer[Int32, MutExternalOrigin](unsafe_from_address=Int(out_idx.unsafe_ptr()))
    var ov_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(out_vals.unsafe_ptr()))

    top_k(vals_ptr, 3, 8, idx_ptr, ov_ptr)

    # Top 3 should be indices 3(0.9), 5(0.8), 7(0.7)
    if Int(out_idx[0]) != 3:
        raise Error("top_k first index should be 3, got " + String(Int(out_idx[0])))
    if Int(out_idx[1]) != 5:
        raise Error("top_k second index should be 5, got " + String(Int(out_idx[1])))
    if Int(out_idx[2]) != 7:
        raise Error("top_k third index should be 7, got " + String(Int(out_idx[2])))
    assert_almost_equal(out_vals[0], 0.9, atol=1e-5)
    assert_almost_equal(out_vals[1], 0.8, atol=1e-5)
    assert_almost_equal(out_vals[2], 0.7, atol=1e-5)
    _ = vals[0]


def test_cpu_backend_rms_norm() raises:
    """CPUBackend.rms_norm matches free function rms_norm."""
    var x = List[Float32](length=4, fill=1.0)
    var w = List[Float32](length=4, fill=2.0)
    var out = List[Float32](length=4, fill=0.0)

    var B = CPUBackend()
    B.rms_norm(out.unsafe_ptr(), x.unsafe_ptr(), w.unsafe_ptr(), 4, 1e-6)

    assert_almost_equal(out[0], 3.0, atol=1e-5)
    assert_almost_equal(out[3], 3.0, atol=1e-5)
    _ = x[0]
    _ = w[0]


def test_cpu_backend_vec_mat_mul() raises:
    """CPUBackend.vec_mat_mul matches free function vec_mat_mul."""
    var x = List[Float32](length=4, fill=1.0)
    var w = List[Float32](length=8, fill=2.0)  # 2x4
    var out = List[Float32](length=2, fill=0.0)

    var B = CPUBackend()
    B.vec_mat_mul(out.unsafe_ptr(), x.unsafe_ptr(), w.unsafe_ptr(), 4, 2)

    assert_almost_equal(out[0], 8.0, atol=1e-5)
    assert_almost_equal(out[1], 8.0, atol=1e-5)
    _ = x[0]
    _ = w[0]


def test_cpu_backend_softmax() raises:
    """CPUBackend.softmax matches free function softmax."""
    var x = List[Float32](length=3, fill=0.0)
    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0

    var B = CPUBackend()
    B.softmax(x.unsafe_ptr(), 3)

    assert_almost_equal(x[0], 0.09003057, atol=1e-5)
    assert_almost_equal(x[1], 0.24472847, atol=1e-5)
    assert_almost_equal(x[2], 0.66524096, atol=1e-5)


def test_cpu_backend_geglu() raises:
    """CPUBackend.geglu matches free function geglu."""
    var gate = List[Float32](length=4, fill=1.0)
    var up = List[Float32](length=4, fill=2.0)
    var out = List[Float32](length=4, fill=0.0)

    var B = CPUBackend()
    B.geglu(out.unsafe_ptr(), gate.unsafe_ptr(), up.unsafe_ptr(), 4)

    assert_almost_equal(out[0], 1.68268, atol=1e-4)
    _ = gate[0]
    _ = up[0]


def test_cpu_backend_softcap() raises:
    var x = List[Float32](length=1, fill=100.0)

    var B = CPUBackend()
    B.softcap(x.unsafe_ptr(), 1, 30.0)

    assert_almost_equal(x[0], 29.9237, atol=1e-3)


def test_cpu_backend_trait_dispatch[B: ComputeBackend](mut backend: B) raises:
    """Verify compile-time trait dispatch with parameterized function."""
    var x = List[Float32](length=4, fill=1.0)
    var w = List[Float32](length=4, fill=0.0)
    var out = List[Float32](length=4, fill=0.0)

    backend.rms_norm(out.unsafe_ptr(), x.unsafe_ptr(), w.unsafe_ptr(), 4, 1e-6)

    # (1+w) = 1.0, x/rms(x) = 1.0 → out ~= 1.0
    assert_almost_equal(out[0], 1.0, atol=1e-5)
    _ = x[0]
    _ = w[0]


def main() raises:
    test_rms_norm()
    test_geglu()
    test_rope_rotate()
    test_vec_mat_mul()
    test_mat_mat_mul()
    test_mat_mat_mul_i8()
    test_softmax()
    test_softcap()
    test_softcap_zero_cap_is_noop()
    test_softcap_default_width_with_tail()
    test_gelu()
    test_average_pool_2d()
    test_top_k()
    test_cpu_backend_rms_norm()
    test_cpu_backend_vec_mat_mul()
    test_cpu_backend_softmax()
    test_cpu_backend_geglu()
    test_cpu_backend_softcap()
    var B = CPUBackend()
    test_cpu_backend_trait_dispatch(B)
    print("Mojo math primitive tests passed!")
