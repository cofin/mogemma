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
from std.gpu import block_idx, thread_idx, block_dim, global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
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
