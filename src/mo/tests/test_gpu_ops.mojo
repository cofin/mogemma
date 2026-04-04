"""Tests for GPU compute kernels in ops_gpu.mojo.

Since CI may not have a GPU, tests verify:
1. Module imports compile correctly
2. Utility functions produce correct results on CPU
3. On GPU systems: kernel launch, data round-trip, and numerical parity with CPU ops
"""

from std.sys import has_accelerator
from std.memory import UnsafePointer
from std.collections import List
from std.testing import assert_almost_equal

from mogemma.ops_gpu import (
    gelu_kernel, geglu_kernel, rope_rotate_kernel,
    ceildiv, optimal_block_size, BLOCK_1D, TILE_BK, TILE_BM, TILE_BN,
)


# ---------------------------------------------------------------------------
# Utility function tests (run on CPU, no GPU needed)
# ---------------------------------------------------------------------------


def test_ceildiv() raises:
    """Test integer ceiling division."""
    if ceildiv(10, 3) != 4:
        raise Error("ceildiv(10, 3) should be 4")
    if ceildiv(9, 3) != 3:
        raise Error("ceildiv(9, 3) should be 3")
    if ceildiv(1, 1) != 1:
        raise Error("ceildiv(1, 1) should be 1")
    if ceildiv(0, 5) != 0:
        raise Error("ceildiv(0, 5) should be 0")
    if ceildiv(256, 256) != 1:
        raise Error("ceildiv(256, 256) should be 1")
    if ceildiv(257, 256) != 2:
        raise Error("ceildiv(257, 256) should be 2")
    print("  test_ceildiv passed")


def test_optimal_block_size() raises:
    """Test power-of-2 block size selection."""
    if optimal_block_size(1) != 32:
        raise Error("optimal_block_size(1) should be 32")
    if optimal_block_size(32) != 32:
        raise Error("optimal_block_size(32) should be 32")
    if optimal_block_size(33) != 64:
        raise Error("optimal_block_size(33) should be 64")
    if optimal_block_size(100) != 128:
        raise Error("optimal_block_size(100) should be 128")
    if optimal_block_size(256) != 256:
        raise Error("optimal_block_size(256) should be 256")
    if optimal_block_size(2000) != 1024:
        raise Error("optimal_block_size(2000) should be 1024")
    print("  test_optimal_block_size passed")


def test_comptime_constants() raises:
    """Verify launch configuration constants have expected values."""
    if BLOCK_1D != 256:
        raise Error("BLOCK_1D should be 256")
    if TILE_BK != 64:
        raise Error("TILE_BK should be 64")
    if TILE_BM != 16:
        raise Error("TILE_BM should be 16")
    if TILE_BN != 16:
        raise Error("TILE_BN should be 16")
    print("  test_comptime_constants passed")


def test_gpu_ops_imports():
    """Verify all GPU kernel functions are importable."""
    print("  test_gpu_ops_imports passed")


# ---------------------------------------------------------------------------
# GPU kernel tests (gated behind has_accelerator)
# ---------------------------------------------------------------------------


def test_gelu_kernel_gpu() raises:
    """Test gelu_kernel produces correct output on GPU."""
    comptime if has_accelerator():
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()
        var size = 4

        # Prepare input: [1.0, 0.0, -1.0, 2.0]
        var x_host = ctx.enqueue_create_host_buffer[DType.float32](size)
        var out_host = ctx.enqueue_create_host_buffer[DType.float32](size)
        ctx.synchronize()
        x_host[0] = 1.0
        x_host[1] = 0.0
        x_host[2] = -1.0
        x_host[3] = 2.0

        var x_dev = ctx.enqueue_create_buffer[DType.float32](size)
        var out_dev = ctx.enqueue_create_buffer[DType.float32](size)
        ctx.enqueue_copy(x_dev, x_host)

        ctx.enqueue_function[gelu_kernel, gelu_kernel](
            out_dev, x_dev, size,
            grid_dim=1, block_dim=size,
        )

        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()

        # gelu(1.0) ≈ 0.8413, gelu(0.0) = 0.0, gelu(-1.0) ≈ -0.1587, gelu(2.0) ≈ 1.9545
        assert_almost_equal(out_host[0], Float32(0.8413), atol=1e-3)
        assert_almost_equal(out_host[1], Float32(0.0), atol=1e-6)
        assert_almost_equal(out_host[2], Float32(-0.1587), atol=1e-3)
        assert_almost_equal(out_host[3], Float32(1.9545), atol=1e-3)
        print("  test_gelu_kernel_gpu passed")
    else:
        print("  SKIP: test_gelu_kernel_gpu (no GPU)")


def test_geglu_kernel_gpu() raises:
    """Test geglu_kernel produces correct output on GPU."""
    comptime if has_accelerator():
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()
        var size = 4

        var gate_host = ctx.enqueue_create_host_buffer[DType.float32](size)
        var up_host = ctx.enqueue_create_host_buffer[DType.float32](size)
        var out_host = ctx.enqueue_create_host_buffer[DType.float32](size)
        ctx.synchronize()

        for i in range(size):
            gate_host[i] = 1.0
            up_host[i] = 2.0

        var gate_dev = ctx.enqueue_create_buffer[DType.float32](size)
        var up_dev = ctx.enqueue_create_buffer[DType.float32](size)
        var out_dev = ctx.enqueue_create_buffer[DType.float32](size)
        ctx.enqueue_copy(gate_dev, gate_host)
        ctx.enqueue_copy(up_dev, up_host)

        ctx.enqueue_function[geglu_kernel, geglu_kernel](
            out_dev, gate_dev, up_dev, size,
            grid_dim=1, block_dim=size,
        )

        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()

        # gelu(1.0) ≈ 0.8413, 0.8413 * 2.0 ≈ 1.6827
        for i in range(size):
            assert_almost_equal(out_host[i], Float32(1.6827), atol=1e-3)
        print("  test_geglu_kernel_gpu passed")
    else:
        print("  SKIP: test_geglu_kernel_gpu (no GPU)")


def test_rope_rotate_kernel_gpu() raises:
    """Test rope_rotate_kernel produces correct output on GPU."""
    comptime if has_accelerator():
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()
        var head_dim = 4
        var half_dim = head_dim // 2

        # vec = [1.0, 0.0, 0.0, 1.0], cos = [0.0, 1.0], sin = [1.0, 0.0]
        var vec_host = ctx.enqueue_create_host_buffer[DType.float32](head_dim)
        var cos_host = ctx.enqueue_create_host_buffer[DType.float32](half_dim)
        var sin_host = ctx.enqueue_create_host_buffer[DType.float32](half_dim)
        ctx.synchronize()

        vec_host[0] = 1.0
        vec_host[1] = 0.0
        vec_host[2] = 0.0
        vec_host[3] = 1.0
        cos_host[0] = 0.0
        cos_host[1] = 1.0
        sin_host[0] = 1.0
        sin_host[1] = 0.0

        var vec_dev = ctx.enqueue_create_buffer[DType.float32](head_dim)
        var cos_dev = ctx.enqueue_create_buffer[DType.float32](half_dim)
        var sin_dev = ctx.enqueue_create_buffer[DType.float32](half_dim)
        ctx.enqueue_copy(vec_dev, vec_host)
        ctx.enqueue_copy(cos_dev, cos_host)
        ctx.enqueue_copy(sin_dev, sin_host)

        ctx.enqueue_function[rope_rotate_kernel, rope_rotate_kernel](
            vec_dev, cos_dev, sin_dev, half_dim,
            grid_dim=1, block_dim=half_dim,
        )

        ctx.enqueue_copy(vec_host, vec_dev)
        ctx.synchronize()

        # [1*0 - 0*1, 0*1 - 1*0, 0*0 + 1*1, 1*1 + 0*0] = [0, 0, 1, 1]
        assert_almost_equal(vec_host[0], Float32(0.0), atol=1e-5)
        assert_almost_equal(vec_host[1], Float32(0.0), atol=1e-5)
        assert_almost_equal(vec_host[2], Float32(1.0), atol=1e-5)
        assert_almost_equal(vec_host[3], Float32(1.0), atol=1e-5)
        print("  test_rope_rotate_kernel_gpu passed")
    else:
        print("  SKIP: test_rope_rotate_kernel_gpu (no GPU)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() raises:
    print("GPU ops tests:")
    test_gpu_ops_imports()
    test_ceildiv()
    test_optimal_block_size()
    test_comptime_constants()
    test_gelu_kernel_gpu()
    test_geglu_kernel_gpu()
    test_rope_rotate_kernel_gpu()
    print("GPU ops tests passed!")
