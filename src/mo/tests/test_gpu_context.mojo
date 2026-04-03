"""Tests for GPUContext struct and GPU memory infrastructure.

Since CI may not have a GPU, tests verify:
1. Module imports compile correctly
2. comptime guard properly gates GPU-only code
3. Struct construction and methods are type-correct
4. On GPU systems: actual buffer allocation and data round-trip
"""

from std.sys import has_accelerator
from std.memory import UnsafePointer
from std.collections import List

from mogemma.gpu_context import GPUContext


def test_gpu_context_imports():
    """Verify GPUContext is importable and types resolve."""
    print("GPUContext module imported successfully")


def test_gpu_context_lifecycle() raises:
    """Test GPUContext creation and cleanup on GPU, or graceful skip on CPU."""
    comptime if has_accelerator():
        var ctx = GPUContext()
        print("GPUContext created on GPU device")
        ctx.cleanup()
        print("GPUContext cleaned up")
    else:
        print("SKIP: test_gpu_context_lifecycle (no GPU)")


def test_allocate_device_buffer() raises:
    """Test DeviceBuffer allocation via GPUContext."""
    comptime if has_accelerator():
        var ctx = GPUContext()
        var buf = ctx.allocate_buffer[DType.float32](1024)
        print("Allocated 1024-element float32 DeviceBuffer")
        _ = buf
        ctx.cleanup()
    else:
        print("SKIP: test_allocate_device_buffer (no GPU)")


def test_allocate_host_buffer() raises:
    """Test HostBuffer (pinned memory) allocation via GPUContext."""
    comptime if has_accelerator():
        var ctx = GPUContext()
        var buf = ctx.allocate_host_buffer[DType.float32](1024)
        print("Allocated 1024-element float32 HostBuffer")
        _ = buf
        ctx.cleanup()
    else:
        print("SKIP: test_allocate_host_buffer (no GPU)")


def test_upload_download_roundtrip() raises:
    """Test host → device → host data round-trip preserves values."""
    comptime if has_accelerator():
        var ctx = GPUContext()

        # Create host buffer with known values
        var host_src = ctx.allocate_host_buffer[DType.float32](16)
        var src_ptr = host_src.unsafe_ptr()
        for i in range(16):
            src_ptr.store(i, Float32(Float64(i) * 3.14))

        # Upload to device
        var dev_buf = ctx.allocate_buffer[DType.float32](16)
        ctx.upload(dev_buf, host_src)
        ctx.sync()

        # Download back
        var host_dst = ctx.allocate_host_buffer[DType.float32](16)
        ctx.download(host_dst, dev_buf)
        ctx.sync()

        # Verify round-trip
        var dst_ptr = host_dst.unsafe_ptr()
        for i in range(16):
            var expected = Float32(Float64(i) * 3.14)
            var actual = dst_ptr.load(i)
            var diff = actual - expected
            if diff < 0:
                diff = -diff
            if diff > 1e-5:
                raise Error("Round-trip mismatch at index " + String(i))
        print("Upload/download round-trip verified (16 elements)")

        _ = dev_buf
        _ = host_src
        _ = host_dst
        ctx.cleanup()
    else:
        print("SKIP: test_upload_download_roundtrip (no GPU)")


def main() raises:
    test_gpu_context_imports()
    test_gpu_context_lifecycle()
    test_allocate_device_buffer()
    test_allocate_host_buffer()
    test_upload_download_roundtrip()
    print("All gpu_context tests passed")
