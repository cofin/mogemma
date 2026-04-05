from std.math import sqrt, erf, exp
from std.memory import UnsafePointer


@always_inline
def geglu[
    nelts: Int = 16
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    gate_ptr: UnsafePointer[Float32, MutAnyOrigin],
    up_ptr: UnsafePointer[Float32, MutAnyOrigin],
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
def gelu[
    nelts: Int = 16
](out_ptr: UnsafePointer[Float32, MutAnyOrigin], x_ptr: UnsafePointer[Float32, MutAnyOrigin], size: Int,):
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
def average_pool_2d(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
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
def rope_rotate[
    nelts: Int = 16
](
    vec_ptr: UnsafePointer[Float32, MutAnyOrigin],
    cos_ptr: UnsafePointer[Float32, MutAnyOrigin],
    sin_ptr: UnsafePointer[Float32, MutAnyOrigin],
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
def vec_mat_mul[
    nelts: Int = 16
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    w_ptr: UnsafePointer[Float32, MutAnyOrigin],  # transposed [out_dim, in_dim]
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
def mat_mat_mul[
    nelts: Int = 16
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [batch_size, in_dim]
    w_ptr: UnsafePointer[Float32, MutAnyOrigin],  # transposed [out_dim, in_dim]
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
def vec_mat_mul_i8[
    nelts: Int = 16
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    w_ptr: UnsafePointer[Int8, MutAnyOrigin],  # transposed [out_dim, in_dim]
    scale_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [1] or [out_dim]
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
def mat_mat_mul_i8[
    nelts: Int = 16
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [batch_size, in_dim]
    w_ptr: UnsafePointer[Int8, MutAnyOrigin],  # transposed [out_dim, in_dim]
    scale_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [1]
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
def rms_norm[
    nelts: Int = 16
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    weight_ptr: UnsafePointer[Float32, MutAnyOrigin],
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
def softmax[nelts: Int = 16](vec_ptr: UnsafePointer[Float32, MutAnyOrigin], size: Int):
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


@always_inline
def top_k(
    values_ptr: UnsafePointer[Float32, MutAnyOrigin],
    k: Int,
    size: Int,
    out_indices_ptr: UnsafePointer[Int32, MutAnyOrigin],
    out_values_ptr: UnsafePointer[Float32, MutAnyOrigin],
):
    """Find the k largest values and their indices from a vector.

    Simple linear scan repeated k times. Output sorted descending.
    """
    for sel in range(k):
        var best_val: Float32 = -1e30
        var best_idx: Int32 = 0
        for i in range(size):
            var already_selected = False
            for j in range(sel):
                if Int(out_indices_ptr.load(j)) == i:
                    already_selected = True
                    break
            if not already_selected and values_ptr.load(i) > best_val:
                best_val = values_ptr.load(i)
                best_idx = Int32(i)
        out_indices_ptr.store(sel, best_idx)
        out_values_ptr.store(sel, best_val)


# ---------------------------------------------------------------------------
# ComputeBackend trait — unified CPU/GPU dispatch contract
# ---------------------------------------------------------------------------


trait ComputeBackend:
    """Dispatch contract for CPU and GPU math kernels.

    Layers parameterize on this trait to avoid code duplication between
    CPU and GPU paths. All pointer arguments use MutAnyOrigin so both
    CPU heap pointers (MutExternalOrigin, which widens implicitly) and
    GPU DeviceBuffer pointers work without casts.
    """

    def vec_mat_mul(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_ptr: UnsafePointer[Float32, MutAnyOrigin],
        w_ptr: UnsafePointer[Float32, MutAnyOrigin],
        in_dim: Int, out_dim: Int,
    ): ...

    def vec_mat_mul_i8(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_ptr: UnsafePointer[Float32, MutAnyOrigin],
        w_ptr: UnsafePointer[Int8, MutAnyOrigin],
        scale_ptr: UnsafePointer[Float32, MutAnyOrigin],
        in_dim: Int, out_dim: Int,
    ): ...

    def mat_mat_mul(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_ptr: UnsafePointer[Float32, MutAnyOrigin],
        w_ptr: UnsafePointer[Float32, MutAnyOrigin],
        batch_size: Int, in_dim: Int, out_dim: Int,
    ): ...

    def rms_norm(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_ptr: UnsafePointer[Float32, MutAnyOrigin],
        weight_ptr: UnsafePointer[Float32, MutAnyOrigin],
        size: Int, eps: Float32,
    ): ...

    def softmax(
        mut self,
        vec_ptr: UnsafePointer[Float32, MutAnyOrigin],
        size: Int,
    ): ...

    def rope_rotate(
        mut self,
        vec_ptr: UnsafePointer[Float32, MutAnyOrigin],
        cos_ptr: UnsafePointer[Float32, MutAnyOrigin],
        sin_ptr: UnsafePointer[Float32, MutAnyOrigin],
        head_dim: Int,
    ): ...

    def geglu(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        gate_ptr: UnsafePointer[Float32, MutAnyOrigin],
        up_ptr: UnsafePointer[Float32, MutAnyOrigin],
        size: Int,
    ): ...

    def gelu(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_ptr: UnsafePointer[Float32, MutAnyOrigin],
        size: Int,
    ): ...

    def copy(
        mut self,
        dst_ptr: UnsafePointer[Float32, MutAnyOrigin],
        src_ptr: UnsafePointer[Float32, MutAnyOrigin],
        size: Int,
    ): ...

    def embed_lookup(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        embed_table_ptr: UnsafePointer[Float32, MutAnyOrigin],
        token_id: Int,
        hidden_size: Int,
        scale: Float32,
    ): ...

    def vector_add(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        a_ptr: UnsafePointer[Float32, MutAnyOrigin],
        b_ptr: UnsafePointer[Float32, MutAnyOrigin],
        size: Int,
    ): ...

    def vector_add_scaled(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        a_ptr: UnsafePointer[Float32, MutAnyOrigin],
        b_ptr: UnsafePointer[Float32, MutAnyOrigin],
        scale: Float32,
        size: Int,
    ): ...

    def average_pool_2d(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_ptr: UnsafePointer[Float32, MutAnyOrigin],
        grid_h: Int, grid_w: Int, hidden_size: Int, kernel: Int,
    ): ...

    def top_k(
        mut self,
        values_ptr: UnsafePointer[Float32, MutAnyOrigin],
        k: Int, size: Int,
        out_indices_ptr: UnsafePointer[Int32, MutAnyOrigin],
        out_values_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ): ...

    def kv_write(
        mut self,
        dst_ptr: UnsafePointer[Float32, MutAnyOrigin],
        src_ptr: UnsafePointer[Float32, MutAnyOrigin],
        kv_size: Int, pos: Int, cache_size: Int,
        layer_offset: Int, is_full: Bool,
    ): ...

    def attention_scores(
        mut self,
        scores_ptr: UnsafePointer[Float32, MutAnyOrigin],
        q_ptr: UnsafePointer[Float32, MutAnyOrigin],
        k_cache_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_heads: Int, num_kv_heads: Int, head_dim: Int,
        valid_len: Int, kv_size: Int, scale: Float32,
    ): ...

    def attention_value_accum(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        scores_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_cache_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_heads: Int, num_kv_heads: Int, head_dim: Int,
        valid_len: Int, kv_size: Int,
    ): ...


# ---------------------------------------------------------------------------
# CPUBackend — delegates to the SIMD-vectorized free functions above
# ---------------------------------------------------------------------------


struct CPUBackend(ComputeBackend):
    """CPU implementation of ComputeBackend.

    Each method delegates to the corresponding @always_inline free function
    defined above. Zero overhead — all calls inline through the trait.
    """

    def __init__(out self):
        pass

    def vec_mat_mul(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_ptr: UnsafePointer[Float32, MutAnyOrigin],
        w_ptr: UnsafePointer[Float32, MutAnyOrigin],
        in_dim: Int, out_dim: Int,
    ):
        vec_mat_mul(out_ptr, x_ptr, w_ptr, in_dim, out_dim)

    def vec_mat_mul_i8(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_ptr: UnsafePointer[Float32, MutAnyOrigin],
        w_ptr: UnsafePointer[Int8, MutAnyOrigin],
        scale_ptr: UnsafePointer[Float32, MutAnyOrigin],
        in_dim: Int, out_dim: Int,
    ):
        vec_mat_mul_i8(out_ptr, x_ptr, w_ptr, scale_ptr, in_dim, out_dim)

    def mat_mat_mul(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_ptr: UnsafePointer[Float32, MutAnyOrigin],
        w_ptr: UnsafePointer[Float32, MutAnyOrigin],
        batch_size: Int, in_dim: Int, out_dim: Int,
    ):
        mat_mat_mul(out_ptr, x_ptr, w_ptr, batch_size, in_dim, out_dim)

    def rms_norm(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_ptr: UnsafePointer[Float32, MutAnyOrigin],
        weight_ptr: UnsafePointer[Float32, MutAnyOrigin],
        size: Int, eps: Float32,
    ):
        rms_norm(out_ptr, x_ptr, weight_ptr, size, eps)

    def softmax(
        mut self,
        vec_ptr: UnsafePointer[Float32, MutAnyOrigin],
        size: Int,
    ):
        softmax(vec_ptr, size)

    def rope_rotate(
        mut self,
        vec_ptr: UnsafePointer[Float32, MutAnyOrigin],
        cos_ptr: UnsafePointer[Float32, MutAnyOrigin],
        sin_ptr: UnsafePointer[Float32, MutAnyOrigin],
        head_dim: Int,
    ):
        rope_rotate(vec_ptr, cos_ptr, sin_ptr, head_dim)

    def geglu(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        gate_ptr: UnsafePointer[Float32, MutAnyOrigin],
        up_ptr: UnsafePointer[Float32, MutAnyOrigin],
        size: Int,
    ):
        geglu(out_ptr, gate_ptr, up_ptr, size)

    def gelu(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_ptr: UnsafePointer[Float32, MutAnyOrigin],
        size: Int,
    ):
        gelu(out_ptr, x_ptr, size)

    def copy(
        mut self,
        dst_ptr: UnsafePointer[Float32, MutAnyOrigin],
        src_ptr: UnsafePointer[Float32, MutAnyOrigin],
        size: Int,
    ):
        for i in range(size):
            dst_ptr.store(i, src_ptr.load(i))

    def embed_lookup(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        embed_table_ptr: UnsafePointer[Float32, MutAnyOrigin],
        token_id: Int,
        hidden_size: Int,
        scale: Float32,
    ):
        var src = embed_table_ptr + token_id * hidden_size
        for i in range(hidden_size):
            out_ptr.store(i, src.load(i) * scale)

    def vector_add(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        a_ptr: UnsafePointer[Float32, MutAnyOrigin],
        b_ptr: UnsafePointer[Float32, MutAnyOrigin],
        size: Int,
    ):
        for i in range(size):
            out_ptr.store(i, a_ptr.load(i) + b_ptr.load(i))

    def vector_add_scaled(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        a_ptr: UnsafePointer[Float32, MutAnyOrigin],
        b_ptr: UnsafePointer[Float32, MutAnyOrigin],
        scale: Float32,
        size: Int,
    ):
        for i in range(size):
            out_ptr.store(i, a_ptr.load(i) + scale * b_ptr.load(i))

    def average_pool_2d(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_ptr: UnsafePointer[Float32, MutAnyOrigin],
        grid_h: Int, grid_w: Int, hidden_size: Int, kernel: Int,
    ):
        average_pool_2d(out_ptr, x_ptr, grid_h, grid_w, hidden_size, kernel)

    def top_k(
        mut self,
        values_ptr: UnsafePointer[Float32, MutAnyOrigin],
        k: Int, size: Int,
        out_indices_ptr: UnsafePointer[Int32, MutAnyOrigin],
        out_values_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ):
        top_k(values_ptr, k, size, out_indices_ptr, out_values_ptr)

    def kv_write(
        mut self,
        dst_ptr: UnsafePointer[Float32, MutAnyOrigin],
        src_ptr: UnsafePointer[Float32, MutAnyOrigin],
        kv_size: Int, pos: Int, cache_size: Int,
        layer_offset: Int, is_full: Bool,
    ):
        var write_pos: Int
        if is_full:
            write_pos = pos
        else:
            write_pos = pos % cache_size
        
        var dst = dst_ptr + layer_offset + write_pos * kv_size
        for i in range(kv_size):
            dst.store(i, src_ptr.load(i))

    def attention_scores(
        mut self,
        scores_ptr: UnsafePointer[Float32, MutAnyOrigin],
        q_ptr: UnsafePointer[Float32, MutAnyOrigin],
        k_cache_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_heads: Int, num_kv_heads: Int, head_dim: Int,
        valid_len: Int, kv_size: Int, scale: Float32,
    ):
        var heads_per_kv = num_heads // num_kv_heads
        for h in range(num_heads):
            var kv_h = h // heads_per_kv
            var q_head = q_ptr + h * head_dim
            var head_scores = scores_ptr + h * valid_len
            for t in range(valid_len):
                var k_head = k_cache_ptr + t * kv_size + kv_h * head_dim
                var score: Float32 = 0.0
                for d in range(head_dim):
                    score += q_head.load(d) * k_head.load(d)
                head_scores.store(t, score * scale)

    def attention_value_accum(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        scores_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_cache_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_heads: Int, num_kv_heads: Int, head_dim: Int,
        valid_len: Int, kv_size: Int,
    ):
        var heads_per_kv = num_heads // num_kv_heads
        for h in range(num_heads):
            var kv_h = h // heads_per_kv
            var out_head = out_ptr + h * head_dim
            var head_probs = scores_ptr + h * valid_len
            for d in range(head_dim):
                out_head.store(d, 0.0)
            for t in range(valid_len):
                var v_head = v_cache_ptr + t * kv_size + kv_h * head_dim
                var prob = head_probs.load(t)
                for d in range(head_dim):
                    out_head.store(d, out_head.load(d) + prob * v_head.load(d))
