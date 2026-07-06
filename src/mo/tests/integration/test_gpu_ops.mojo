"""Tests for GPU compute kernels in ops_gpu.mojo.

Since CI may not have a GPU, tests verify:
1. Module imports compile correctly
2. Utility functions produce correct results on CPU
3. On GPU systems: kernel launch, data round-trip, and numerical parity with CPU ops
"""

from std.sys import has_accelerator
from mogemma.gpu_context import has_usable_gpu
from std.memory import UnsafePointer
from std.collections import List
from std.testing import assert_almost_equal

from mogemma.ops_gpu import (
    gelu_kernel,
    geglu_kernel,
    rope_rotate_kernel,
    softmax_kernel,
    softmax_strided_kernel,
    rms_norm_kernel,
    vec_mat_mul_kernel,
    mat_mat_mul_kernel,
    vec_mat_mul_i8_kernel,
    average_pool_2d_kernel,
    top_k_kernel,
    GPUBackend,
    ceildiv,
    optimal_block_size,
    BLOCK_1D,
    TILE_BK,
    TILE_BM,
    TILE_BN,
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
    comptime if has_usable_gpu():
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

        ctx.enqueue_function[gelu_kernel](
            out_dev,
            x_dev,
            size,
            grid_dim=1,
            block_dim=size,
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
    comptime if has_usable_gpu():
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

        ctx.enqueue_function[geglu_kernel](
            out_dev,
            gate_dev,
            up_dev,
            size,
            grid_dim=1,
            block_dim=size,
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
    comptime if has_usable_gpu():
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

        ctx.enqueue_function[rope_rotate_kernel](
            vec_dev,
            cos_dev,
            sin_dev,
            half_dim,
            grid_dim=1,
            block_dim=half_dim,
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


def test_softmax_kernel_gpu() raises:
    """Test softmax_kernel produces correct output on GPU (small vector)."""
    comptime if has_usable_gpu():
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()
        var size = 3

        var x_host = ctx.enqueue_create_host_buffer[DType.float32](size)
        ctx.synchronize()
        x_host[0] = 1.0
        x_host[1] = 2.0
        x_host[2] = 3.0

        var x_dev = ctx.enqueue_create_buffer[DType.float32](size)
        ctx.enqueue_copy(x_dev, x_host)

        # Use BLOCK_SIZE=32 (smallest power-of-2 >= 3)
        ctx.enqueue_function[softmax_kernel[32]](
            x_dev,
            size,
            grid_dim=1,
            block_dim=32,
        )

        ctx.enqueue_copy(x_host, x_dev)
        ctx.synchronize()

        # Expected: [0.09003057, 0.24472847, 0.66524096]
        assert_almost_equal(x_host[0], Float32(0.09003057), atol=1e-5)
        assert_almost_equal(x_host[1], Float32(0.24472847), atol=1e-5)
        assert_almost_equal(x_host[2], Float32(0.66524096), atol=1e-5)
        print("  test_softmax_kernel_gpu passed")
    else:
        print("  SKIP: test_softmax_kernel_gpu (no GPU)")


def test_softmax_strided_kernel_gpu() raises:
    """Test softmax_strided_kernel for larger vectors."""
    comptime if has_usable_gpu():
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()
        var size = 3

        var x_host = ctx.enqueue_create_host_buffer[DType.float32](size)
        ctx.synchronize()
        x_host[0] = 1.0
        x_host[1] = 2.0
        x_host[2] = 3.0

        var x_dev = ctx.enqueue_create_buffer[DType.float32](size)
        ctx.enqueue_copy(x_dev, x_host)

        # Use strided variant even for small size to test it
        ctx.enqueue_function[softmax_strided_kernel[32]](
            x_dev,
            size,
            grid_dim=1,
            block_dim=32,
        )

        ctx.enqueue_copy(x_host, x_dev)
        ctx.synchronize()

        assert_almost_equal(x_host[0], Float32(0.09003057), atol=1e-5)
        assert_almost_equal(x_host[1], Float32(0.24472847), atol=1e-5)
        assert_almost_equal(x_host[2], Float32(0.66524096), atol=1e-5)
        print("  test_softmax_strided_kernel_gpu passed")
    else:
        print("  SKIP: test_softmax_strided_kernel_gpu (no GPU)")


def test_rms_norm_kernel_gpu() raises:
    """Test rms_norm_kernel with Gemma's (1+w) scaling on GPU."""
    comptime if has_usable_gpu():
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()
        var size = 4

        var x_host = ctx.enqueue_create_host_buffer[DType.float32](size)
        var w_host = ctx.enqueue_create_host_buffer[DType.float32](size)
        var out_host = ctx.enqueue_create_host_buffer[DType.float32](size)
        ctx.synchronize()

        # x = [1, 1, 1, 1], w = [2, 2, 2, 2]
        # rms = sqrt(4/4 + 1e-6) = 1.0, inv_rms = 1.0
        # out = 1.0 * 1.0 * (1 + 2) = 3.0
        for i in range(size):
            x_host[i] = 1.0
            w_host[i] = 2.0

        var x_dev = ctx.enqueue_create_buffer[DType.float32](size)
        var w_dev = ctx.enqueue_create_buffer[DType.float32](size)
        var out_dev = ctx.enqueue_create_buffer[DType.float32](size)
        ctx.enqueue_copy(x_dev, x_host)
        ctx.enqueue_copy(w_dev, w_host)

        ctx.enqueue_function[rms_norm_kernel[32]](
            out_dev,
            x_dev,
            w_dev,
            size,
            Float32(1e-6),
            grid_dim=1,
            block_dim=32,
        )

        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()

        for i in range(size):
            assert_almost_equal(out_host[i], Float32(3.0), atol=1e-5)
        print("  test_rms_norm_kernel_gpu passed")
    else:
        print("  SKIP: test_rms_norm_kernel_gpu (no GPU)")


def test_vec_mat_mul_kernel_gpu() raises:
    """Test vec_mat_mul_kernel with shared memory tiling on GPU."""
    comptime if has_usable_gpu():
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()
        var in_dim = 4
        var out_dim = 2

        # x = [1, 1, 1, 1], w = [[2,2,2,2],[2,2,2,2]] -> out = [8, 8]
        var x_host = ctx.enqueue_create_host_buffer[DType.float32](in_dim)
        var w_host = ctx.enqueue_create_host_buffer[DType.float32](out_dim * in_dim)
        var out_host = ctx.enqueue_create_host_buffer[DType.float32](out_dim)
        ctx.synchronize()

        for i in range(in_dim):
            x_host[i] = 1.0
        for i in range(out_dim * in_dim):
            w_host[i] = 2.0

        var x_dev = ctx.enqueue_create_buffer[DType.float32](in_dim)
        var w_dev = ctx.enqueue_create_buffer[DType.float32](out_dim * in_dim)
        var out_dev = ctx.enqueue_create_buffer[DType.float32](out_dim)
        ctx.enqueue_copy(x_dev, x_host)
        ctx.enqueue_copy(w_dev, w_host)

        var grid = ceildiv(out_dim, BLOCK_1D)
        ctx.enqueue_function[vec_mat_mul_kernel](
            out_dev,
            x_dev,
            w_dev,
            in_dim,
            out_dim,
            grid_dim=grid,
            block_dim=BLOCK_1D,
            shared_mem_bytes=TILE_BK * 4,
        )

        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()

        assert_almost_equal(out_host[0], Float32(8.0), atol=1e-5)
        assert_almost_equal(out_host[1], Float32(8.0), atol=1e-5)
        print("  test_vec_mat_mul_kernel_gpu passed")
    else:
        print("  SKIP: test_vec_mat_mul_kernel_gpu (no GPU)")


def test_mat_mat_mul_kernel_gpu() raises:
    """Test mat_mat_mul_kernel 2D tiled matmul on GPU."""
    comptime if has_usable_gpu():
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()
        var batch = 2
        var in_dim = 4
        var out_dim = 2

        var x_host = ctx.enqueue_create_host_buffer[DType.float32](batch * in_dim)
        var w_host = ctx.enqueue_create_host_buffer[DType.float32](out_dim * in_dim)
        var out_host = ctx.enqueue_create_host_buffer[DType.float32](batch * out_dim)
        ctx.synchronize()

        for i in range(batch * in_dim):
            x_host[i] = 1.0
        for i in range(out_dim * in_dim):
            w_host[i] = 2.0

        var x_dev = ctx.enqueue_create_buffer[DType.float32](batch * in_dim)
        var w_dev = ctx.enqueue_create_buffer[DType.float32](out_dim * in_dim)
        var out_dev = ctx.enqueue_create_buffer[DType.float32](batch * out_dim)
        ctx.enqueue_copy(x_dev, x_host)
        ctx.enqueue_copy(w_dev, w_host)

        var grid_x = ceildiv(out_dim, TILE_BN)
        var grid_y = ceildiv(batch, TILE_BM)
        var shared_bytes = (TILE_BM * TILE_BK + TILE_BK * TILE_BN) * 4
        ctx.enqueue_function[mat_mat_mul_kernel](
            out_dev,
            x_dev,
            w_dev,
            batch,
            in_dim,
            out_dim,
            grid_dim=(grid_x, grid_y),
            block_dim=(TILE_BN, TILE_BM),
            shared_mem_bytes=shared_bytes,
        )

        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()

        # 1.0 * 2.0 * 4 = 8.0 for each element
        for i in range(batch * out_dim):
            assert_almost_equal(out_host[i], Float32(8.0), atol=1e-5)
        print("  test_mat_mat_mul_kernel_gpu passed")
    else:
        print("  SKIP: test_mat_mat_mul_kernel_gpu (no GPU)")


def test_vec_mat_mul_i8_kernel_gpu() raises:
    """Test vec_mat_mul_i8_kernel quantized matmul on GPU."""
    comptime if has_usable_gpu():
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()
        var in_dim = 4
        var out_dim = 2

        var x_host = ctx.enqueue_create_host_buffer[DType.float32](in_dim)
        var w_host = ctx.enqueue_create_host_buffer[DType.int8](out_dim * in_dim)
        var scale_host = ctx.enqueue_create_host_buffer[DType.float32](1)
        var out_host = ctx.enqueue_create_host_buffer[DType.float32](out_dim)
        ctx.synchronize()

        for i in range(in_dim):
            x_host[i] = 1.0
        for i in range(out_dim * in_dim):
            w_host[i] = Int8(10)
        scale_host[0] = 0.2  # dequant: 10 * 0.2 = 2.0

        var x_dev = ctx.enqueue_create_buffer[DType.float32](in_dim)
        var w_dev = ctx.enqueue_create_buffer[DType.int8](out_dim * in_dim)
        var scale_dev = ctx.enqueue_create_buffer[DType.float32](1)
        var out_dev = ctx.enqueue_create_buffer[DType.float32](out_dim)
        ctx.enqueue_copy(x_dev, x_host)
        ctx.enqueue_copy(w_dev, w_host)
        ctx.enqueue_copy(scale_dev, scale_host)

        var grid = ceildiv(out_dim, BLOCK_1D)
        ctx.enqueue_function[vec_mat_mul_i8_kernel](
            out_dev,
            x_dev,
            w_dev,
            scale_dev,
            in_dim,
            out_dim,
            grid_dim=grid,
            block_dim=BLOCK_1D,
            shared_mem_bytes=TILE_BK * 4,
        )

        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()

        # 1.0 * 2.0 * 4 = 8.0
        assert_almost_equal(out_host[0], Float32(8.0), atol=1e-5)
        assert_almost_equal(out_host[1], Float32(8.0), atol=1e-5)
        print("  test_vec_mat_mul_i8_kernel_gpu passed")
    else:
        print("  SKIP: test_vec_mat_mul_i8_kernel_gpu (no GPU)")


def test_average_pool_2d_kernel_gpu() raises:
    """Test average_pool_2d_kernel on GPU."""
    comptime if has_usable_gpu():
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()
        var grid_h = 6
        var grid_w = 6
        var hidden_size = 2
        var kernel = 3
        var out_h = grid_h // kernel  # 2
        var out_w = grid_w // kernel  # 2
        var in_size = grid_h * grid_w * hidden_size  # 72
        var out_size = out_h * out_w * hidden_size  # 8

        var x_host = ctx.enqueue_create_host_buffer[DType.float32](in_size)
        var out_host = ctx.enqueue_create_host_buffer[DType.float32](out_size)
        ctx.synchronize()

        # All 1.0 → averages should all be 1.0
        for i in range(in_size):
            x_host[i] = 1.0

        var x_dev = ctx.enqueue_create_buffer[DType.float32](in_size)
        var out_dev = ctx.enqueue_create_buffer[DType.float32](out_size)
        ctx.enqueue_copy(x_dev, x_host)

        ctx.enqueue_function[average_pool_2d_kernel](
            out_dev,
            x_dev,
            out_h,
            out_w,
            grid_w,
            hidden_size,
            kernel,
            grid_dim=out_h * out_w,
            block_dim=hidden_size,
        )

        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()

        for i in range(out_size):
            assert_almost_equal(out_host[i], Float32(1.0), atol=1e-5)
        print("  test_average_pool_2d_kernel_gpu passed")
    else:
        print("  SKIP: test_average_pool_2d_kernel_gpu (no GPU)")


def test_top_k_kernel_gpu() raises:
    """Test top_k_kernel on GPU."""
    comptime if has_usable_gpu():
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()
        var size = 8
        var k = 3

        var vals_host = ctx.enqueue_create_host_buffer[DType.float32](size)
        var idx_host = ctx.enqueue_create_host_buffer[DType.int32](k)
        var ov_host = ctx.enqueue_create_host_buffer[DType.float32](k)
        ctx.synchronize()

        # [0.1, 0.5, 0.3, 0.9, 0.2, 0.8, 0.4, 0.7]
        vals_host[0] = 0.1
        vals_host[1] = 0.5
        vals_host[2] = 0.3
        vals_host[3] = 0.9
        vals_host[4] = 0.2
        vals_host[5] = 0.8
        vals_host[6] = 0.4
        vals_host[7] = 0.7

        var vals_dev = ctx.enqueue_create_buffer[DType.float32](size)
        var idx_dev = ctx.enqueue_create_buffer[DType.int32](k)
        var ov_dev = ctx.enqueue_create_buffer[DType.float32](k)
        ctx.enqueue_copy(vals_dev, vals_host)

        ctx.enqueue_function[top_k_kernel](
            vals_dev,
            k,
            size,
            idx_dev,
            ov_dev,
            grid_dim=1,
            block_dim=32,
        )

        ctx.enqueue_copy(idx_host, idx_dev)
        ctx.enqueue_copy(ov_host, ov_dev)
        ctx.synchronize()

        # Top 3: index 3 (0.9), index 5 (0.8), index 7 (0.7)
        assert_almost_equal(ov_host[0], Float32(0.9), atol=1e-5)
        assert_almost_equal(ov_host[1], Float32(0.8), atol=1e-5)
        assert_almost_equal(ov_host[2], Float32(0.7), atol=1e-5)
        print("  test_top_k_kernel_gpu passed")
    else:
        print("  SKIP: test_top_k_kernel_gpu (no GPU)")


def test_gpu_backend_launch_gelu() raises:
    """Test GPUBackend.launch_gelu convenience method."""
    comptime if has_usable_gpu():
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()
        var size = 4

        var x_host = ctx.enqueue_create_host_buffer[DType.float32](size)
        var out_host = ctx.enqueue_create_host_buffer[DType.float32](size)
        ctx.synchronize()
        for i in range(size):
            x_host[i] = 1.0

        var x_dev = ctx.enqueue_create_buffer[DType.float32](size)
        var out_dev = ctx.enqueue_create_buffer[DType.float32](size)
        ctx.enqueue_copy(x_dev, x_host)

        var backend = GPUBackend(rebind[UnsafePointer[DeviceContext, MutAnyOrigin]](0))  # DUMMY for parsing
        backend.gelu(out_dev.unsafe_ptr(), x_dev.unsafe_ptr(), size)

        ctx.enqueue_copy(out_host, out_dev)
        ctx.synchronize()

        assert_almost_equal(out_host[0], Float32(0.8413), atol=1e-3)
        print("  test_gpu_backend_launch_gelu passed")
    else:
        print("  SKIP: test_gpu_backend_launch_gelu (no GPU)")


def test_gpu_backend_imports():
    """Verify GPUBackend is importable."""
    print("  test_gpu_backend_imports passed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() raises:
    print("GPU ops tests:")
    test_gpu_ops_imports()
    test_ceildiv()
    test_optimal_block_size()
    test_comptime_constants()
    test_gpu_backend_imports()
    test_gelu_kernel_gpu()
    test_geglu_kernel_gpu()
    test_rope_rotate_kernel_gpu()
    test_softmax_kernel_gpu()
    test_softmax_strided_kernel_gpu()
    test_rms_norm_kernel_gpu()
    test_vec_mat_mul_kernel_gpu()
    test_mat_mat_mul_kernel_gpu()
    test_vec_mat_mul_i8_kernel_gpu()
    test_average_pool_2d_kernel_gpu()
    test_top_k_kernel_gpu()
    test_gpu_backend_launch_gelu()
    print("GPU ops tests passed!")
