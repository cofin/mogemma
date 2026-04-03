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

from mogemma.gpu_context import GPUContext, WeightStage


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


def test_weight_stage_imports():
    """Verify WeightStage is importable."""
    print("WeightStage type imported successfully")


def test_weight_stage_creation() raises:
    """Test WeightStage allocation on GPU, skip on CPU."""
    comptime if has_accelerator():
        var ctx = GPUContext()
        var stage = WeightStage(ctx, capacity=4096)
        print("WeightStage created with capacity 4096")
        _ = stage
        ctx.cleanup()
    else:
        print("SKIP: test_weight_stage_creation (no GPU)")


def test_weight_stage_upload_tensor() raises:
    """Test uploading a tensor via WeightStage and verifying data round-trip."""
    comptime if has_accelerator():
        from mogemma.model import TensorInfo

        var ctx = GPUContext()
        var stage = WeightStage(ctx, capacity=64)

        # Create a fake tensor in host memory (simulating mmap'd safetensors)
        var fake_data = List[Float32](length=12, fill=0.0)
        for i in range(12):
            fake_data[i] = Float32(Float64(i) + 1.0)
        var fake_ptr = UnsafePointer[Float32, MutExternalOrigin](
            unsafe_from_address=Int(fake_data.unsafe_ptr())
        )

        # Create TensorInfo pointing to fake data (3x4 matrix)
        var tensor = TensorInfo(Int(fake_data.unsafe_ptr()), 3, 4)

        # Upload tensor to device staging
        var dev_ptr = stage.upload_tensor(ctx, tensor)
        ctx.sync()

        # Download back to verify
        var host_out = ctx.allocate_host_buffer[DType.float32](12)
        ctx.download(host_out, stage.device_buf)
        ctx.sync()

        var out_ptr = host_out.unsafe_ptr()
        for i in range(12):
            var expected = Float32(Float64(i) + 1.0)
            var actual = out_ptr.load(i)
            var diff = actual - expected
            if diff < 0:
                diff = -diff
            if diff > 1e-5:
                raise Error("WeightStage tensor round-trip mismatch at " + String(i))
        print("WeightStage upload_tensor round-trip verified (12 elements)")

        _ = stage
        _ = fake_data
        _ = host_out
        ctx.cleanup()
    else:
        print("SKIP: test_weight_stage_upload_tensor (no GPU)")


def main() raises:
    test_gpu_context_imports()
    test_gpu_context_lifecycle()
    test_allocate_device_buffer()
    test_allocate_host_buffer()
    test_upload_download_roundtrip()
    test_weight_stage_imports()
    test_weight_stage_creation()
    test_weight_stage_upload_tensor()
    print("All gpu_context tests passed")
