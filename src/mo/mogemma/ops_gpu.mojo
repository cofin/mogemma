from memory import UnsafePointer

@always_inline
fn geglu_gpu(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    gate_ptr: UnsafePointer[Float32, MutExternalOrigin],
    up_ptr: UnsafePointer[Float32, MutExternalOrigin],
    size: Int
):
    # Stub: just zeroes output to fail parity tests
    for i in range(size):
        out_ptr.store(i, 0.0)

@always_inline
fn rope_rotate_gpu(
    vec_ptr: UnsafePointer[Float32, MutExternalOrigin],
    cos_ptr: UnsafePointer[Float32, MutExternalOrigin],
    sin_ptr: UnsafePointer[Float32, MutExternalOrigin],
    head_dim: Int
):
    # Stub: zero out vector to fail
    for i in range(head_dim):
        vec_ptr.store(i, 0.0)

@always_inline
fn vec_mat_mul_gpu(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],
    w_ptr: UnsafePointer[Float32, MutExternalOrigin],
    in_dim: Int,
    out_dim: Int
):
    # Stub: zero out
    for i in range(out_dim):
        out_ptr.store(i, 0.0)

@always_inline
fn rms_norm_gpu(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],
    weight_ptr: UnsafePointer[Float32, MutExternalOrigin],
    size: Int,
    eps: Float32 = 1e-6
):
    # Stub: zero out
    for i in range(size):
        out_ptr.store(i, 0.0)

@always_inline
fn softmax_gpu(
    vec_ptr: UnsafePointer[Float32, MutExternalOrigin],
    size: Int
):
    # Stub: zero out
    for i in range(size):
        vec_ptr.store(i, 0.0)
