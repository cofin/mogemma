"""GPU compute kernels for mogemma.

Provides GPU implementations of all math ops in ops.mojo. Each kernel is
a standalone function launched via DeviceContext.enqueue_function. All GPU
code is gated behind `comptime if has_accelerator()` so this module compiles
on CPU-only systems.

Kernel categories:
- Element-wise: gelu_gpu, geglu_gpu, rope_rotate_gpu (thread-per-element)
- Reduction: softmax_gpu, rms_norm_gpu (warp/block-level reductions)
- Matmul: vec_mat_mul_gpu, mat_mat_mul_gpu (tiled, shared memory)
- Specialized: average_pool_2d_gpu, top_k_gpu
"""

from std.sys import has_accelerator
from std.gpu import block_idx, thread_idx, block_dim, global_idx, warp_id, lane_id
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.sync import barrier, syncwarp
from std.gpu.primitives.warp import sum as warp_sum, max as warp_max, broadcast as warp_broadcast
from std.gpu.primitives.block import sum as block_sum, max as block_max
from std.gpu.memory import AddressSpace, external_memory
from std.memory import UnsafePointer
from std.math import sqrt, erf, exp


# ---------------------------------------------------------------------------
# Launch configuration utilities (Task 2.13, placed here for use by kernels)
# ---------------------------------------------------------------------------

comptime BLOCK_1D: Int = 256
comptime TILE_BK: Int = 64
comptime TILE_BM: Int = 16
comptime TILE_BN: Int = 16


@always_inline
def ceildiv(n: Int, d: Int) -> Int:
    """Integer ceiling division: ceildiv(n, d) = ceil(n / d)."""
    return (n + d - 1) // d


@always_inline
def optimal_block_size(size: Int) -> Int:
    """Return the smallest power-of-2 block size in [32, 1024] that covers size."""
    var block = 32
    while block < size and block < 1024:
        block *= 2
    return block


# ---------------------------------------------------------------------------
# Phase 2: Element-wise GPU kernels (Task 2.3, 2.4)
# ---------------------------------------------------------------------------


def gelu_kernel(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """GPU kernel: GELU activation, one thread per element.

    Computes 0.5 * x * (1 + erf(x / sqrt(2))) for each element.
    Launch: grid_dim = ceildiv(size, BLOCK_1D), block_dim = BLOCK_1D
    """
    var tid = global_idx.x
    if tid < size:
        var x = x_ptr[tid]
        var sqrt_2: Float32 = 1.4142135623730951
        out_ptr[tid] = 0.5 * x * (1.0 + erf(x / sqrt_2))


def geglu_kernel(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    gate_ptr: UnsafePointer[Float32, MutAnyOrigin],
    up_ptr: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """GPU kernel: fused Gate-GELU-Up activation, one thread per element.

    Computes gelu(gate) * up for each element.
    Launch: grid_dim = ceildiv(size, BLOCK_1D), block_dim = BLOCK_1D
    """
    var tid = global_idx.x
    if tid < size:
        var gate = gate_ptr[tid]
        var up = up_ptr[tid]
        var sqrt_2: Float32 = 1.4142135623730951
        var gelu_gate = 0.5 * gate * (1.0 + erf(gate / sqrt_2))
        out_ptr[tid] = gelu_gate * up


def rope_rotate_kernel(
    vec_ptr: UnsafePointer[Float32, MutAnyOrigin],
    cos_ptr: UnsafePointer[Float32, MutAnyOrigin],
    sin_ptr: UnsafePointer[Float32, MutAnyOrigin],
    half_dim: Int,
):
    """GPU kernel: RoPE rotation, one thread per dimension pair.

    Rotates vec in-place: [x1, x2] → [x1*c - x2*s, x2*c + x1*s]
    Launch: grid_dim = 1, block_dim = half_dim (typically 64-256)
    """
    var tid = global_idx.x
    if tid < half_dim:
        var x1 = vec_ptr[tid]
        var x2 = vec_ptr[tid + half_dim]
        var c = cos_ptr[tid]
        var s = sin_ptr[tid]
        vec_ptr[tid] = x1 * c - x2 * s
        vec_ptr[tid + half_dim] = x2 * c + x1 * s


# ---------------------------------------------------------------------------
# Phase 3: Reduction GPU kernels (Task 2.5, 2.6)
# ---------------------------------------------------------------------------


def softmax_kernel[
    BLOCK_SIZE: Int
](
    vec_ptr: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """GPU kernel: in-place softmax with block-level reductions.

    Uses block_max and block_sum for the two-pass algorithm:
    1. Find max value across all elements
    2. Compute exp(x - max) and sum
    3. Normalize by dividing by sum

    Handles sizes up to BLOCK_SIZE. Each thread processes one element.
    Launch: grid_dim = 1, block_dim = BLOCK_SIZE

    BLOCK_SIZE must be a compile-time power-of-2, typically
    optimal_block_size(size) clamped to [32, 1024].
    """
    var tid = thread_idx.x

    # Load value or -inf for padding threads
    var val: Float32 = -1e30
    if tid < size:
        val = vec_ptr[tid]

    # Pass 1: block-wide max
    var max_val = block_max[block_size=BLOCK_SIZE](val)

    # Pass 2: exp and block-wide sum
    var exp_val: Float32 = 0.0
    if tid < size:
        exp_val = exp(val - max_val)
    var sum_exp = block_sum[block_size=BLOCK_SIZE](exp_val)

    # Pass 3: normalize
    if tid < size:
        vec_ptr[tid] = exp_val / sum_exp


def softmax_strided_kernel[
    BLOCK_SIZE: Int
](
    vec_ptr: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """GPU kernel: in-place softmax for vectors larger than BLOCK_SIZE.

    Each thread handles multiple elements via strided access, then
    contributes partial results to block reductions.
    Launch: grid_dim = 1, block_dim = BLOCK_SIZE
    """
    var tid = thread_idx.x

    # Pass 1: strided max
    var local_max: Float32 = -1e30
    var i = tid
    while i < size:
        var val = vec_ptr[i]
        if val > local_max:
            local_max = val
        i += BLOCK_SIZE
    var max_val = block_max[block_size=BLOCK_SIZE](local_max)

    # Pass 2: strided exp + sum
    var local_sum: Float32 = 0.0
    i = tid
    while i < size:
        var e = exp(vec_ptr[i] - max_val)
        vec_ptr[i] = e  # store exp in-place
        local_sum += e
        i += BLOCK_SIZE
    var sum_exp = block_sum[block_size=BLOCK_SIZE](local_sum)

    # Pass 3: strided normalize
    var inv_sum = 1.0 / sum_exp
    i = tid
    while i < size:
        vec_ptr[i] = vec_ptr[i] * inv_sum
        i += BLOCK_SIZE


def rms_norm_kernel[
    BLOCK_SIZE: Int
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    weight_ptr: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
    eps: Float32,
):
    """GPU kernel: RMS normalization with Gemma's (1+w) scaling.

    Uses block_sum for the sum-of-squares reduction, then each thread
    scales its assigned elements.

    Computes: out = x / rms(x) * (1 + w)
    where rms(x) = sqrt(mean(x^2) + eps)

    Launch: grid_dim = 1, block_dim = BLOCK_SIZE
    """
    var tid = thread_idx.x

    # Strided accumulation of sum of squares
    var partial_sq: Float32 = 0.0
    var i = tid
    while i < size:
        var val = x_ptr[i]
        partial_sq += val * val
        i += BLOCK_SIZE

    # Block-wide sum of squares
    var total_sq = block_sum[block_size=BLOCK_SIZE](partial_sq)
    var inv_rms = 1.0 / sqrt(total_sq / Float32(size) + eps)

    # Strided scaling with (1 + w)
    i = tid
    while i < size:
        out_ptr[i] = x_ptr[i] * inv_rms * (1.0 + weight_ptr[i])
        i += BLOCK_SIZE


# ---------------------------------------------------------------------------
# Phase 4: Matmul GPU kernels (Task 2.7, 2.8, 2.9)
# ---------------------------------------------------------------------------


def vec_mat_mul_kernel(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    w_ptr: UnsafePointer[Float32, MutAnyOrigin],
    in_dim: Int,
    out_dim: Int,
):
    """GPU kernel: tiled vector-matrix multiply with shared memory.

    Computes out[o] = sum_i(x[i] * w[o, i]) where w is transposed [out_dim, in_dim].
    Uses shared memory to tile the input vector x so all threads in a block
    reuse the same loaded tile, avoiding redundant global memory reads.

    Each thread computes one output element.
    Launch: grid_dim = ceildiv(out_dim, BLOCK_1D), block_dim = BLOCK_1D,
            shared_mem_bytes = TILE_BK * sizeof(Float32)
    """
    var tid = block_idx.x * BLOCK_1D + thread_idx.x
    if tid >= out_dim:
        return

    var acc: Float32 = 0.0
    var w_row = w_ptr + tid * in_dim

    # Shared memory tile for input vector x
    var x_shared = external_memory[
        SIMD[DType.float32, 1],
        address_space=AddressSpace.SHARED,
        alignment=4,
    ]()

    var tile_start = 0
    while tile_start < in_dim:
        # Cooperative load: first TILE_BK threads load x tile
        if thread_idx.x < TILE_BK and tile_start + thread_idx.x < in_dim:
            x_shared[thread_idx.x] = x_ptr[tile_start + thread_idx.x]
        barrier()

        # Each thread dots its weight row tile against shared x tile
        var tile_end = TILE_BK
        if tile_start + tile_end > in_dim:
            tile_end = in_dim - tile_start
        for k in range(tile_end):
            acc += w_row[tile_start + k] * x_shared[k]
        barrier()

        tile_start += TILE_BK

    out_ptr[tid] = acc


def mat_mat_mul_kernel(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    w_ptr: UnsafePointer[Float32, MutAnyOrigin],
    batch_size: Int,
    in_dim: Int,
    out_dim: Int,
):
    """GPU kernel: 2D tiled matrix multiply with shared memory.

    Computes out[b, o] = sum_i(x[b, i] * w[o, i]) for batched inputs.
    Uses BM×BK and BK×BN shared memory tiles loaded cooperatively.
    w is stored transposed [out_dim, in_dim].

    Launch: grid_dim = (ceildiv(out_dim, TILE_BN), ceildiv(batch_size, TILE_BM)),
            block_dim = (TILE_BN, TILE_BM),
            shared_mem_bytes = (TILE_BM*TILE_BK + TILE_BK*TILE_BN) * sizeof(Float32)
    """
    var row = block_idx.y * TILE_BM + thread_idx.y  # batch index
    var col = block_idx.x * TILE_BN + thread_idx.x  # output index

    if row >= batch_size or col >= out_dim:
        return

    var acc: Float32 = 0.0

    # Shared memory region: x_tile [BM, BK] then w_tile [BK, BN]
    var shared = external_memory[
        SIMD[DType.float32, 1],
        address_space=AddressSpace.SHARED,
        alignment=4,
    ]()
    comptime x_tile_size: Int = TILE_BM * TILE_BK
    comptime w_tile_offset: Int = x_tile_size

    var num_tiles = ceildiv(in_dim, TILE_BK)
    for tile in range(num_tiles):
        var k_base = tile * TILE_BK

        # Load x tile [BM, BK]: each thread loads one element
        if k_base + thread_idx.x < in_dim and row < batch_size:
            shared[thread_idx.y * TILE_BK + thread_idx.x] = x_ptr[row * in_dim + k_base + thread_idx.x]
        else:
            shared[thread_idx.y * TILE_BK + thread_idx.x] = 0.0

        # Load w tile [BK, BN]: w is [out_dim, in_dim], need w[col, k_base+ty]
        if k_base + thread_idx.y < in_dim and col < out_dim:
            shared[w_tile_offset + thread_idx.y * TILE_BN + thread_idx.x] = w_ptr[col * in_dim + k_base + thread_idx.y]
        else:
            shared[w_tile_offset + thread_idx.y * TILE_BN + thread_idx.x] = 0.0

        barrier()

        # Partial dot product over tile
        for k in range(TILE_BK):
            if k_base + k < in_dim:
                acc += shared[thread_idx.y * TILE_BK + k] * shared[w_tile_offset + k * TILE_BN + thread_idx.x]
        barrier()

    out_ptr[row * out_dim + col] = acc


def vec_mat_mul_i8_kernel(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    w_ptr: UnsafePointer[Int8, MutAnyOrigin],
    scale_ptr: UnsafePointer[Float32, MutAnyOrigin],
    in_dim: Int,
    out_dim: Int,
):
    """GPU kernel: quantized vector-matrix multiply (int8 weights).

    Same tiling as vec_mat_mul_kernel but loads int8 weights, casts to
    float32 in the inner loop, and applies per-tensor scale after reduction.

    Launch: grid_dim = ceildiv(out_dim, BLOCK_1D), block_dim = BLOCK_1D,
            shared_mem_bytes = TILE_BK * sizeof(Float32)
    """
    var tid = block_idx.x * BLOCK_1D + thread_idx.x
    if tid >= out_dim:
        return

    var acc: Float32 = 0.0
    var w_row = w_ptr + tid * in_dim

    var x_shared = external_memory[
        SIMD[DType.float32, 1],
        address_space=AddressSpace.SHARED,
        alignment=4,
    ]()

    var tile_start = 0
    while tile_start < in_dim:
        if thread_idx.x < TILE_BK and tile_start + thread_idx.x < in_dim:
            x_shared[thread_idx.x] = x_ptr[tile_start + thread_idx.x]
        barrier()

        var tile_end = TILE_BK
        if tile_start + tile_end > in_dim:
            tile_end = in_dim - tile_start
        for k in range(tile_end):
            acc += Float32(w_row[tile_start + k]) * x_shared[k]
        barrier()

        tile_start += TILE_BK

    out_ptr[tid] = acc * scale_ptr[0]


# ---------------------------------------------------------------------------
# Phase 5: Specialized GPU kernels (Task 2.10, 2.11)
# ---------------------------------------------------------------------------


def average_pool_2d_kernel(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    out_h: Int,
    out_w: Int,
    grid_w: Int,
    hidden_size: Int,
    kernel: Int,
):
    """GPU kernel: average pooling over kernel×kernel blocks.

    One block per output spatial position, threads parallelize across
    the hidden dimension. For SigLIP: kernel=3, 9 inputs per output.

    Input layout: [grid_h * grid_w, hidden_size]
    Output layout: [out_h * out_w, hidden_size]

    Launch: grid_dim = out_h * out_w, block_dim = min(hidden_size, 1024)
    """
    var out_idx = block_idx.x  # output spatial position
    var d = thread_idx.x  # hidden dimension index

    if d >= hidden_size:
        return

    var oh = out_idx // out_w
    var ow = out_idx % out_w
    var inv_block = 1.0 / Float32(kernel * kernel)

    var acc: Float32 = 0.0
    for kh in range(kernel):
        for kw in range(kernel):
            var in_r = oh * kernel + kh
            var in_c = ow * kernel + kw
            var in_idx = (in_r * grid_w + in_c) * hidden_size + d
            acc += x_ptr[in_idx]

    out_ptr[out_idx * hidden_size + d] = acc * inv_block


def top_k_kernel(
    values_ptr: UnsafePointer[Float32, MutAnyOrigin],
    k: Int,
    size: Int,
    out_indices_ptr: UnsafePointer[Int32, MutAnyOrigin],
    out_values_ptr: UnsafePointer[Float32, MutAnyOrigin],
):
    """GPU kernel: find k largest values using warp-level max.

    Single-warp kernel for small k (typically 8) and small size
    (typically 128 for MoE expert selection). Warp lanes each check
    a subset of values, warp_max finds the global winner, repeat k times.

    Launch: grid_dim = 1, block_dim = 32 (one warp)
    """
    var tid = thread_idx.x  # lane ID within the warp
    comptime WARP_SIZE: Int = 32

    # Each lane handles a stripe of the input
    for sel in range(k):
        var best_val: Float32 = -1e30
        var best_idx: Int32 = -1

        # Each lane scans its assigned elements
        var i = tid
        while i < size:
            # Check if already selected
            var already = False
            for j in range(sel):
                if Int(out_indices_ptr[j]) == i:
                    already = True
                    break
            if not already and values_ptr[i] > best_val:
                best_val = values_ptr[i]
                best_idx = Int32(i)
            i += WARP_SIZE

        # Warp-level max to find global winner
        # We need the index too, so encode val+idx and reduce
        # Use warp_max on value, then broadcast the winner's index
        var global_max = warp_max(best_val)

        # The lane with the winning value writes its index
        # If multiple lanes have the same max, first one wins
        var winner_idx: Int32 = -1
        if best_val == global_max and best_idx >= 0:
            winner_idx = best_idx
        # Broadcast from the first matching lane
        var winner = warp_max(winner_idx)

        # Lane 0 writes the result
        if tid == 0:
            out_indices_ptr[sel] = winner
            out_values_ptr[sel] = global_max
