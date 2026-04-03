from std.math import sqrt, erf, exp
from std.memory import UnsafePointer


@always_inline
fn geglu[
    nelts: Int = 16
](
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    gate_ptr: UnsafePointer[Float32, MutExternalOrigin],
    up_ptr: UnsafePointer[Float32, MutExternalOrigin],
    size: Int,
):
    """Applies the GEGLU activation function element-wise.

    Reads from the gate and up tensors, computes the GELU of the gate, multiplies it by the up value, and writes the result to the output tensor.
    """
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
fn gelu[
    nelts: Int = 16
](
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],
    size: Int,
):
    """Applies the standard GELU activation function element-wise.

    Computes 0.5 * x * (1 + erf(x / sqrt(2))) for each element.
    """
    var i = 0
    var sqrt_2: Float32 = 1.4142135623730951
    while i <= size - nelts:
        var x = x_ptr.load[width=nelts](i)
        out_ptr.store(i, 0.5 * x * (1.0 + erf(x / sqrt_2)))
        i += nelts

    while i < size:
        var x = x_ptr.load(i)
        out_ptr.store(i, 0.5 * x * (1.0 + erf(x / sqrt_2)))
        i += 1


@always_inline
fn average_pool_2d(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],
    grid_h: Int,
    grid_w: Int,
    hidden_size: Int,
    kernel: Int,
):
    """Average pooling over non-overlapping kernel×kernel blocks on a 2D token grid.

    Input layout: [grid_h * grid_w, hidden_size] (row-major by grid position).
    Output layout: [(grid_h/kernel) * (grid_w/kernel), hidden_size].
    """
    var out_h = grid_h // kernel
    var out_w = grid_w // kernel
    var block_size = kernel * kernel
    var inv_block = 1.0 / Float32(block_size)

    for oh in range(out_h):
        for ow in range(out_w):
            var out_idx = (oh * out_w + ow) * hidden_size
            # Zero output
            for d in range(hidden_size):
                out_ptr.store(out_idx + d, 0.0)
            # Sum over kernel×kernel block
            for kh in range(kernel):
                for kw in range(kernel):
                    var in_r = oh * kernel + kh
                    var in_c = ow * kernel + kw
                    var in_idx = (in_r * grid_w + in_c) * hidden_size
                    for d in range(hidden_size):
                        out_ptr.store(
                            out_idx + d,
                            out_ptr.load(out_idx + d) + x_ptr.load(in_idx + d),
                        )
            # Average
            for d in range(hidden_size):
                out_ptr.store(out_idx + d, out_ptr.load(out_idx + d) * inv_block)


@always_inline
fn rope_rotate[
    nelts: Int = 16
](
    vec_ptr: UnsafePointer[Float32, MutExternalOrigin],
    cos_ptr: UnsafePointer[Float32, MutExternalOrigin],
    sin_ptr: UnsafePointer[Float32, MutExternalOrigin],
    head_dim: Int,
):
    """Applies Rotary Positional Embedding (RoPE) to an attention head vector in place.

    Rotates the input vector's values using the provided cosine and sine frequency tensors.
    """
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
    w_ptr: UnsafePointer[Float32, MutExternalOrigin],  # transposed [out_dim, in_dim]
    in_dim: Int,
    out_dim: Int,
):
    """Performs a vector-matrix multiplication.

    Multiplies the input vector by a transposed weight matrix and writes the resulting vector to the output tensor.
    """
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
fn mat_mat_mul[
    nelts: Int = 16
](
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],  # [batch_size, in_dim]
    w_ptr: UnsafePointer[Float32, MutExternalOrigin],  # transposed [out_dim, in_dim]
    batch_size: Int,
    in_dim: Int,
    out_dim: Int,
):
    """Performs a batched matrix-matrix multiplication.

    Multiplies the batched input matrix by a transposed weight matrix and writes the resulting matrix to the output tensor.
    """
    for b in range(batch_size):
        vec_mat_mul[nelts](
            out_ptr + b * out_dim,
            x_ptr + b * in_dim,
            w_ptr,
            in_dim,
            out_dim,
        )


@always_inline
fn vec_mat_mul_i8[
    nelts: Int = 16
](
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],
    w_ptr: UnsafePointer[Int8, MutExternalOrigin],  # transposed [out_dim, in_dim]
    scale_ptr: UnsafePointer[Float32, MutExternalOrigin],  # [1] or [out_dim]
    in_dim: Int,
    out_dim: Int,
):
    """Performs a vector-matrix multiplication using symmetric int8 weights.

    Multiplies the input vector by a transposed 8-bit weight matrix, dequantizes the accumulations using the provided scale, and writes the resulting float32 vector to the output tensor.
    """
    for o in range(out_dim):
        var acc: Float32 = 0.0
        var i = 0
        var w_row_ptr = w_ptr + o * in_dim

        while i <= in_dim - nelts:
            var x_val = x_ptr.load[width=nelts](i)
            # load as Int8, then cast to Float32 for math
            var w_val_i8 = w_row_ptr.load[width=nelts](i)
            var w_val = w_val_i8.cast[DType.float32]()
            acc += (x_val * w_val).reduce_add()
            i += nelts

        while i < in_dim:
            var x_val = x_ptr.load(i)
            var w_val = Float32(w_row_ptr.load(i))
            acc += x_val * w_val
            i += 1

        # We assume per-tensor scale for now, where scale_ptr has size 1.
        # If it were per-channel, it would be scale_ptr.load(o).
        var scale = scale_ptr.load(0)
        out_ptr.store(o, acc * scale)


@always_inline
fn mat_mat_mul_i8[
    nelts: Int = 16
](
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],  # [batch_size, in_dim]
    w_ptr: UnsafePointer[Int8, MutExternalOrigin],  # transposed [out_dim, in_dim]
    scale_ptr: UnsafePointer[Float32, MutExternalOrigin],  # [1]
    batch_size: Int,
    in_dim: Int,
    out_dim: Int,
):
    """Performs a batched matrix-matrix multiplication using symmetric int8 weights."""
    for b in range(batch_size):
        vec_mat_mul_i8[nelts](
            out_ptr + b * out_dim,
            x_ptr + b * in_dim,
            w_ptr,
            scale_ptr,
            in_dim,
            out_dim,
        )


@always_inline
fn rms_norm[
    nelts: Int = 16
](
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],
    weight_ptr: UnsafePointer[Float32, MutExternalOrigin],
    size: Int,
    eps: Float32 = 1e-6,
):
    """Applies Root Mean Square (RMS) Normalization.

    Normalizes the input vector and scales it using the provided weight tensor, storing the result in the output tensor.
    """
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

    var mean_sq = sum_sq / Float32(size)
    var inv_rms = 1.0 / sqrt(mean_sq + eps)

    i = 0
    while i <= size - nelts:
        var val = x_ptr.load[width=nelts](i)
        var w = weight_ptr.load[width=nelts](i)
        var res = val * inv_rms * (1.0 + w)
        out_ptr.store(i, res)
        i += nelts

    while i < size:
        var val = x_ptr.load(i)
        var w = weight_ptr.load(i)
        var res = val * inv_rms * (1.0 + w)
        out_ptr.store(i, res)
        i += 1


@always_inline
fn softmax[nelts: Int = 16](vec_ptr: UnsafePointer[Float32, MutExternalOrigin], size: Int):
    """Applies the softmax operation to a vector in place.

    Transforms the input values into a normalized probability distribution.
    """
    # Find max
    var max_val: Float32 = -1e9
    var i = 0
    while i < size:
        var val = vec_ptr.load(i)
        if val > max_val:
            max_val = val
        i += 1

    # Exp and sum
    var sum_exp: Float32 = 0.0
    i = 0
    while i <= size - nelts:
        var val = vec_ptr.load[width=nelts](i)
        var e = exp(val - max_val)
        sum_exp += e.reduce_add()
        vec_ptr.store(i, e)
        i += nelts

    while i < size:
        var val = vec_ptr.load(i)
        var e = exp(val - max_val)
        sum_exp += e
        vec_ptr.store(i, e)
        i += 1

    # Normalize
    var inv_sum = 1.0 / sum_exp
    i = 0
    while i <= size - nelts:
        var e = vec_ptr.load[width=nelts](i)
        vec_ptr.store(i, e * inv_sum)
        i += nelts

    while i < size:
        var e = vec_ptr.load(i)
        vec_ptr.store(i, e * inv_sum)
        i += 1
