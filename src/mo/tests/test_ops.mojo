from collections import List
from mogemma.ops import rms_norm, geglu, rope_rotate, vec_mat_mul, softmax
from memory import UnsafePointer
from testing import assert_almost_equal

fn test_rms_norm() raises:
    # 4 elements: all 1.0. Mean sq = 1.0, inv_rms ~= 1.0
    var x = List[Float32](length=4, fill=1.0)
    var w = List[Float32](length=4, fill=2.0)
    var out = List[Float32](length=4, fill=0.0)
    
    # We must cast the pointers
    var x_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(x.unsafe_ptr()))
    var w_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(w.unsafe_ptr()))
    var out_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(out.unsafe_ptr()))
    
    rms_norm[2](out_ptr, x_ptr, w_ptr, 4)
    
    assert_almost_equal(out[0], 2.0, atol=1e-5)
    assert_almost_equal(out[1], 2.0, atol=1e-5)
    assert_almost_equal(out[2], 2.0, atol=1e-5)
    assert_almost_equal(out[3], 2.0, atol=1e-5)
    _ = x[0]
    _ = w[0]

fn test_geglu() raises:
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

fn test_rope_rotate() raises:
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

fn test_vec_mat_mul() raises:
    var x = List[Float32](length=4, fill=1.0)
    var w = List[Float32](length=8, fill=2.0) # 2x4
    var out = List[Float32](length=2, fill=0.0)
    
    var x_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(x.unsafe_ptr()))
    var w_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(w.unsafe_ptr()))
    var out_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(out.unsafe_ptr()))
    
    vec_mat_mul[2](out_ptr, x_ptr, w_ptr, 4, 2)
    
    assert_almost_equal(out[0], 8.0, atol=1e-5)
    assert_almost_equal(out[1], 8.0, atol=1e-5)
    _ = x[0]
    _ = w[0]

fn main() raises:
    test_rms_norm()
    test_geglu()
    test_rope_rotate()
    test_vec_mat_mul()
    test_softmax()
    print("Mojo math primitive tests passed!")

fn test_softmax() raises:
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
