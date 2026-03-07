from memory import UnsafePointer
from mogemma.ops import geglu, rope_rotate, vec_mat_mul, rms_norm, softmax


@always_inline
fn geglu_gpu(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    gate_ptr: UnsafePointer[Float32, MutExternalOrigin],
    up_ptr: UnsafePointer[Float32, MutExternalOrigin],
    size: Int,
):
    # Polyfill with CPU implementation until PTX/Max GPU integration
    geglu[1](out_ptr, gate_ptr, up_ptr, size)


@always_inline
fn rope_rotate_gpu(
    vec_ptr: UnsafePointer[Float32, MutExternalOrigin],
    cos_ptr: UnsafePointer[Float32, MutExternalOrigin],
    sin_ptr: UnsafePointer[Float32, MutExternalOrigin],
    head_dim: Int,
):
    # Polyfill
    rope_rotate[1](vec_ptr, cos_ptr, sin_ptr, head_dim)


@always_inline
fn vec_mat_mul_gpu(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],
    w_ptr: UnsafePointer[Float32, MutExternalOrigin],
    in_dim: Int,
    out_dim: Int,
):
    # Polyfill
    vec_mat_mul[1](out_ptr, x_ptr, w_ptr, in_dim, out_dim)


@always_inline
fn rms_norm_gpu(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],
    weight_ptr: UnsafePointer[Float32, MutExternalOrigin],
    size: Int,
    eps: Float32 = 1e-6,
):
    # Polyfill
    rms_norm[1](out_ptr, x_ptr, weight_ptr, size, eps)


@always_inline
fn softmax_gpu(vec_ptr: UnsafePointer[Float32, MutExternalOrigin], size: Int):
    # Polyfill
    softmax[1](vec_ptr, size)
