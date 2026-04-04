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
