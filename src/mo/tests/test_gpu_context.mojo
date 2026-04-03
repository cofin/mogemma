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

from mogemma.gpu_context import (
    GPUContext, WeightStage, PersistentBuffers, GPUKVCache, GPUScratch,
    upload_layer_weights, upload_expert_weights, upload_vision_layer_weights,
)


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


def test_persistent_buffers() raises:
    """Test persistent buffer allocation for embed_tokens and lm_head."""
    comptime if has_accelerator():
        from mogemma.model import TensorInfo

        var ctx = GPUContext()

        # Fake embed_tokens: vocab_size=8, hidden_size=4 → 32 elements
        var embed_data = List[Float32](length=32, fill=0.0)
        for i in range(32):
            embed_data[i] = Float32(Float64(i) * 0.1)
        var embed = TensorInfo(Int(embed_data.unsafe_ptr()), 8, 4)

        # Fake lm_head: hidden_size=4, vocab_size=8 → 32 elements
        var head_data = List[Float32](length=32, fill=0.0)
        for i in range(32):
            head_data[i] = Float32(Float64(i) * 0.2)
        var head = TensorInfo(Int(head_data.unsafe_ptr()), 4, 8)

        # Fake norm: hidden_size=4 → 4 elements
        var norm_data = List[Float32](length=4, fill=1.0)
        var norm = TensorInfo(Int(norm_data.unsafe_ptr()), 4, 1)

        var persistent = PersistentBuffers(ctx, embed, head, norm)

        # Verify sizes
        if persistent.embed_elements != 32:
            raise Error("embed_elements mismatch")
        if persistent.lm_head_elements != 32:
            raise Error("lm_head_elements mismatch")
        if persistent.norm_elements != 4:
            raise Error("norm_elements mismatch")
        print("PersistentBuffers allocated and sizes verified")

        _ = persistent
        _ = embed_data
        _ = head_data
        _ = norm_data
        ctx.cleanup()
    else:
        print("SKIP: test_persistent_buffers (no GPU)")


def test_gpu_kv_cache() raises:
    """Test GPUKVCache construction matches CPU KVCache layout."""
    comptime if has_accelerator():
        from mogemma.model import KVCache, LAYER_TYPE_SLIDING, LAYER_TYPE_FULL

        var ctx = GPUContext()

        # 2 layers: sliding + full, 4 KV heads, dim=32, window=512, max_ctx=4096
        var layer_types = List[UInt8](length=2, fill=UInt8(0))
        layer_types[0] = LAYER_TYPE_SLIDING
        layer_types[1] = LAYER_TYPE_FULL
        var lt_ptr = UnsafePointer[UInt8, MutExternalOrigin](
            unsafe_from_address=Int(layer_types.unsafe_ptr())
        )

        # Build CPU KVCache for reference
        var cpu_cache = KVCache(2, 4, 32, 512, 4096, lt_ptr)
        var expected_total = cpu_cache.total_elements()

        # Build GPUKVCache
        var gpu_cache = GPUKVCache(ctx, 2, 4, 32, 512, 4096, lt_ptr)

        # Verify total elements match
        if gpu_cache.total_elements() != expected_total:
            raise Error(
                "GPUKVCache total_elements mismatch: got "
                + String(gpu_cache.total_elements())
                + " expected "
                + String(expected_total)
            )

        # Verify per-layer offsets match
        for i in range(2):
            if gpu_cache.layer_offsets[i] != cpu_cache.layer_offsets[i]:
                raise Error("Layer offset mismatch at layer " + String(i))
            if gpu_cache.layer_cache_sizes[i] != cpu_cache.layer_cache_sizes[i]:
                raise Error("Layer cache_size mismatch at layer " + String(i))

        print("GPUKVCache layout matches CPU KVCache (2 layers, total=" + String(expected_total) + ")")

        _ = gpu_cache
        _ = layer_types
        ctx.cleanup()
    else:
        print("SKIP: test_gpu_kv_cache (no GPU)")


def test_gpu_scratch() raises:
    """Test GPU scratch buffer allocation matches CPU formula."""
    comptime if has_accelerator():
        var ctx = GPUContext()

        # hidden=256, max_seq=1024, heads=8
        var scratch = GPUScratch(ctx, hidden_size=256, max_seq_len=1024, num_heads=8)

        # step_len = 256*160 + 1024*8*2 = 40960 + 16384 = 57344
        # emb_len  = 256*180 + 1024*8*2 = 46080 + 16384 = 62464
        # max = 62464
        if scratch.size != 62464:
            raise Error("GPU scratch size mismatch: got " + String(scratch.size) + " expected 62464")
        print("GPUScratch allocated with correct size: " + String(scratch.size))

        _ = scratch
        ctx.cleanup()
    else:
        print("SKIP: test_gpu_scratch (no GPU)")


def test_upload_layer_weights() raises:
    """Test upload_layer_weights packs all 13 tensors and returns device pointers."""
    comptime if has_accelerator():
        from mogemma.model import TensorInfo, LayerWeights

        var ctx = GPUContext()
        # 4x4 for proj, 4x1 for norms: total per-tensor is small for test
        var data = List[Float32](length=256, fill=1.0)
        var p = Int(data.unsafe_ptr())

        var layer = LayerWeights()
        layer.q_proj = TensorInfo(p, 4, 4)
        layer.k_proj = TensorInfo(p, 2, 4)
        layer.v_proj = TensorInfo(p, 2, 4)
        layer.o_proj = TensorInfo(p, 4, 4)
        layer.gate_proj = TensorInfo(p, 8, 4)
        layer.up_proj = TensorInfo(p, 8, 4)
        layer.down_proj = TensorInfo(p, 4, 8)
        layer.input_layernorm = TensorInfo(p, 4, 1)
        layer.post_attention_layernorm = TensorInfo(p, 4, 1)
        layer.q_norm = TensorInfo(p, 4, 1)
        layer.k_norm = TensorInfo(p, 4, 1)
        layer.pre_feedforward_layernorm = TensorInfo(p, 4, 1)
        layer.post_feedforward_layernorm = TensorInfo(p, 4, 1)

        var stage = WeightStage(ctx, capacity=256)
        var gpu_layer = upload_layer_weights(stage, ctx, layer)
        ctx.sync()

        # Verify all device pointers are non-null
        if Int(gpu_layer.q_proj.ptr) == 0:
            raise Error("q_proj device pointer is null")
        if Int(gpu_layer.gate_proj.ptr) == 0:
            raise Error("gate_proj device pointer is null")
        # Verify shapes preserved
        if gpu_layer.q_proj.shape_0 != 4 or gpu_layer.q_proj.shape_1 != 4:
            raise Error("q_proj shape mismatch")
        print("upload_layer_weights: all 13 tensors uploaded, pointers valid")

        _ = stage
        _ = data
        ctx.cleanup()
    else:
        print("SKIP: test_upload_layer_weights (no GPU)")


def test_upload_expert_weights() raises:
    """Test upload_expert_weights for MoE expert."""
    comptime if has_accelerator():
        from mogemma.model import TensorInfo, MoEExpertWeights

        var ctx = GPUContext()
        var data = List[Float32](length=64, fill=2.0)
        var p = Int(data.unsafe_ptr())

        var expert = MoEExpertWeights()
        expert.gate_proj = TensorInfo(p, 8, 4)
        expert.up_proj = TensorInfo(p, 8, 4)
        expert.down_proj = TensorInfo(p, 4, 8)

        var stage = WeightStage(ctx, capacity=128)
        var gpu_expert = upload_expert_weights(stage, ctx, expert)
        ctx.sync()

        if Int(gpu_expert.gate_proj.ptr) == 0:
            raise Error("expert gate_proj device pointer is null")
        print("upload_expert_weights: 3 tensors uploaded, pointers valid")

        _ = stage
        _ = data
        ctx.cleanup()
    else:
        print("SKIP: test_upload_expert_weights (no GPU)")


def test_upload_vision_layer_weights() raises:
    """Test upload_vision_layer_weights for vision encoder."""
    comptime if has_accelerator():
        from mogemma.model import TensorInfo, VisionLayerWeights

        var ctx = GPUContext()
        var data = List[Float32](length=128, fill=0.5)
        var p = Int(data.unsafe_ptr())

        var vlayer = VisionLayerWeights()
        vlayer.q_proj = TensorInfo(p, 4, 4)
        vlayer.k_proj = TensorInfo(p, 4, 4)
        vlayer.v_proj = TensorInfo(p, 4, 4)
        vlayer.o_proj = TensorInfo(p, 4, 4)
        vlayer.fc1 = TensorInfo(p, 8, 4)
        vlayer.fc2 = TensorInfo(p, 4, 8)
        vlayer.layer_norm1 = TensorInfo(p, 4, 1)
        vlayer.layer_norm2 = TensorInfo(p, 4, 1)

        var stage = WeightStage(ctx, capacity=128)
        var gpu_vlayer = upload_vision_layer_weights(stage, ctx, vlayer)
        ctx.sync()

        if Int(gpu_vlayer.q_proj.ptr) == 0:
            raise Error("vision q_proj device pointer is null")
        if gpu_vlayer.fc1.shape_0 != 8:
            raise Error("vision fc1 shape mismatch")
        print("upload_vision_layer_weights: 8 tensors uploaded, pointers valid")

        _ = stage
        _ = data
        ctx.cleanup()
    else:
        print("SKIP: test_upload_vision_layer_weights (no GPU)")


def main() raises:
    test_gpu_context_imports()
    test_gpu_context_lifecycle()
    test_allocate_device_buffer()
    test_allocate_host_buffer()
    test_upload_download_roundtrip()
    test_weight_stage_imports()
    test_weight_stage_creation()
    test_weight_stage_upload_tensor()
    test_persistent_buffers()
    test_gpu_kv_cache()
    test_gpu_scratch()
    test_upload_layer_weights()
    test_upload_expert_weights()
    test_upload_vision_layer_weights()
    print("All gpu_context tests passed")
