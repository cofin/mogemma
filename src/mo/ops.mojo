from math import sqrt, erf
from memory import UnsafePointer

@always_inline
fn geglu[
    nelts: Int = 16
](
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    gate_ptr: UnsafePointer[Float32, MutExternalOrigin],
    up_ptr: UnsafePointer[Float32, MutExternalOrigin],
    size: Int
):
    var i = 0
    var sqrt_2: Float32 = 1.4142135623730951
    while i <= size - nelts:
        var gate = gate_ptr.load[width=nelts](i)
        var up = up_ptr.load[width=nelts](i)
        
        # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        var gelu_gate = 0.5 * gate * (1.0 + erf(gate / sqrt_2))
        
        out_ptr.store(i, gelu_gate * up)
        i += nelts
        
    while i < size:
        var gate = gate_ptr.load(i)
        var up = up_ptr.load(i)
        var gelu_gate = 0.5 * gate * (1.0 + erf(gate / sqrt_2))
        out_ptr.store(i, gelu_gate * up)
        i += 1

@always_inline
fn rope_rotate[
    nelts: Int = 16
](
    vec_ptr: UnsafePointer[Float32, MutExternalOrigin],
    cos_ptr: UnsafePointer[Float32, MutExternalOrigin],
    sin_ptr: UnsafePointer[Float32, MutExternalOrigin],
    head_dim: Int
):
    # Applies RoPE to a vector of length `head_dim`
    # Assumes half-and-half rotation where x = [x1, x2]
    # rotated(x) = [x1 * cos - x2 * sin, x2 * cos + x1 * sin]
    var half_dim = head_dim // 2
    var i = 0
    
    while i <= half_dim - nelts:
        var x1 = vec_ptr.load[width=nelts](i)
        var x2 = vec_ptr.load[width=nelts](i + half_dim)
        
        var c = cos_ptr.load[width=nelts](i)
        var s = sin_ptr.load[width=nelts](i)
        
        vec_ptr.store(i, x1 * c - x2 * s)
        vec_ptr.store(i + half_dim, x2 * c + x1 * s)
        i += nelts
        
    while i < half_dim:
        var x1 = vec_ptr.load(i)
        var x2 = vec_ptr.load(i + half_dim)
        var c = cos_ptr.load(i)
        var s = sin_ptr.load(i)
        
        vec_ptr.store(i, x1 * c - x2 * s)
        vec_ptr.store(i + half_dim, x2 * c + x1 * s)
        i += 1

@always_inline
fn vec_mat_mul[
    nelts: Int = 16
](
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],
    w_ptr: UnsafePointer[Float32, MutExternalOrigin], # transposed [out_dim, in_dim]
    in_dim: Int,
    out_dim: Int
):
    for o in range(out_dim):
        var acc: Float32 = 0.0
        var i = 0
        var w_row_ptr = w_ptr + o * in_dim
        
        while i <= in_dim - nelts:
            var x_val = x_ptr.load[width=nelts](i)
            var w_val = w_row_ptr.load[width=nelts](i)
            acc += (x_val * w_val).reduce_add()
            i += nelts
            
        while i < in_dim:
            var x_val = x_ptr.load(i)
            var w_val = w_row_ptr.load(i)
            acc += x_val * w_val
            i += 1
            
        out_ptr.store(o, acc)

@always_inline
fn rms_norm[
    nelts: Int = 16
](
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],
    weight_ptr: UnsafePointer[Float32, MutExternalOrigin],
    size: Int,
    eps: Float32 = 1e-6
):
    var sum_sq: Float32 = 0.0
    var i = 0
    while i <= size - nelts:
        var val = x_ptr.load[width=nelts](i)
        var sq = val * val
        sum_sq += sq.reduce_add()
        i += nelts
        
    while i < size:
        var val = x_ptr.load(i)
        sum_sq += val * val
        i += 1
        
    var mean_sq = sum_sq / size
    var inv_rms = 1.0 / sqrt(mean_sq + eps)
    
    i = 0
    while i <= size - nelts:
        var val = x_ptr.load[width=nelts](i)
        var w = weight_ptr.load[width=nelts](i)
        var res = val * inv_rms * w
        out_ptr.store(i, res)
        i += nelts
        
    while i < size:
        var val = x_ptr.load(i)
        var w = weight_ptr.load(i)
        var res = val * inv_rms * w
        out_ptr.store(i, res)
        i += 1
