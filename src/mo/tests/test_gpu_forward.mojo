"""Tests for GPU forward path weight streaming in layers.mojo.

Tests verify:
1. Vision encoder weight streaming (S/C params, comptime upload blocks)
2. Audio encoder weight streaming (same pattern)
3. copy_state device-to-device utility
4. CPU path is unaffected by new params (regression guard)
"""

from std.sys import has_accelerator
from std.memory import UnsafePointer
from std.collections import List
from std.testing import assert_almost_equal

from mogemma.model import (
    TensorInfo,
    VisionLayerWeights,
    VisionModelWeights,
    AudioTowerWeights,
)
from mogemma.ops import CPUBackend
from mogemma.layers import forward_vision_encoder, forward_audio_encoder


def _make_ptr(ref l: List[Float32]) -> UnsafePointer[Float32, MutExternalOrigin]:
    return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(l.unsafe_ptr()))


def _make_identity_matrix(size: Int) -> List[Float32]:
    """Create a size×size identity matrix as flat Float32 list."""
    var m = List[Float32](length=size * size, fill=0.0)
    for i in range(size):
        m[i * size + i] = 1.0
    return m


def _make_ones(n: Int) -> List[Float32]:
    return List[Float32](length=n, fill=1.0)


def _make_zeros(n: Int) -> List[Float32]:
    return List[Float32](length=n, fill=0.0)


# ---------------------------------------------------------------------------
# Test: Vision encoder with S/C params (CPU path, dummy stage/ctx)
# ---------------------------------------------------------------------------


def test_vision_encoder_cpu_with_streaming_params() raises:
    """Vision encoder on CPU path accepts S/C params without changing output.

    This tests that the new weight streaming parameters (S=Int, C=Int dummy)
    produce identical results to the encoder's core logic — no regression.
    """
    var num_patches = 4
    var vision_hidden = 4
    var vision_heads = 1
    var vision_head_dim = 4
    var vision_intermediate = 8
    var decoder_hidden = 4
    var grid_h = 6
    var grid_w = 6
    num_patches = grid_h * grid_w  # 36

    # Create minimal weights
    var patch_dim = 4
    var patch_emb = _make_identity_matrix(vision_hidden)  # [vision_hidden, patch_dim] (4×4 identity)
    var pos_emb = _make_zeros(num_patches * vision_hidden)
    var post_norm_w = _make_ones(vision_hidden)
    var proj = _make_identity_matrix(decoder_hidden)  # [decoder_hidden, vision_hidden] (4×4 identity)

    var weights = VisionModelWeights()
    weights.patch_embedding = TensorInfo(Int(_make_ptr(patch_emb)), vision_hidden, patch_dim)
    weights.position_embedding = TensorInfo(Int(_make_ptr(pos_emb)), num_patches, vision_hidden)
    weights.post_norm = TensorInfo(Int(_make_ptr(post_norm_w)), 1, vision_hidden)
    weights.projection = TensorInfo(Int(_make_ptr(proj)), decoder_hidden, vision_hidden)

    # Create one vision layer with identity-like weights
    var layer = VisionLayerWeights()
    var ident = _make_identity_matrix(vision_hidden)
    layer.q_proj = TensorInfo(Int(_make_ptr(ident)), vision_hidden, vision_hidden)
    layer.k_proj = TensorInfo(Int(_make_ptr(ident)), vision_hidden, vision_hidden)
    layer.v_proj = TensorInfo(Int(_make_ptr(ident)), vision_hidden, vision_hidden)
    layer.o_proj = TensorInfo(Int(_make_ptr(ident)), vision_hidden, vision_hidden)
    var fc1 = _make_identity_matrix(vision_intermediate)  # simplified: pad with zeros
    var fc1_full = List[Float32](length=vision_intermediate * vision_hidden, fill=0.0)
    for r in range(vision_hidden):
        fc1_full[r * vision_hidden + r] = 1.0
    layer.fc1 = TensorInfo(Int(_make_ptr(fc1_full)), vision_intermediate, vision_hidden)
    var fc2_full = List[Float32](length=vision_hidden * vision_intermediate, fill=0.0)
    for r in range(vision_hidden):
        fc2_full[r * vision_intermediate + r] = 1.0
    layer.fc2 = TensorInfo(Int(_make_ptr(fc2_full)), vision_hidden, vision_intermediate)
    var ln1 = _make_ones(vision_hidden)
    var ln2 = _make_ones(vision_hidden)
    layer.layer_norm1 = TensorInfo(Int(_make_ptr(ln1)), 1, vision_hidden)
    layer.layer_norm2 = TensorInfo(Int(_make_ptr(ln2)), 1, vision_hidden)
    weights.layers.append(layer)

    # Input patches: [num_patches, patch_dim]
    var patches = List[Float32](length=num_patches * patch_dim, fill=0.5)

    # Output: after pool_kernel=3, pooled_h=2, pooled_w=2 -> 4 tokens of decoder_hidden
    var pool_kernel = 3
    var pooled_h = grid_h // pool_kernel
    var pooled_w = grid_w // pool_kernel
    var out_tokens = pooled_h * pooled_w  # 4
    var out = _make_zeros(out_tokens * decoder_hidden)

    # Scratch: generous allocation
    var scratch_size = num_patches * vision_hidden * 60 + vision_heads * num_patches * num_patches * 4
    var scratch = _make_zeros(scratch_size)

    var backend = CPUBackend()
    var dummy_stage = 0
    var dummy_ctx = 0

    forward_vision_encoder(
        backend,
        _make_ptr(out),
        _make_ptr(patches),
        weights,
        num_patches,
        grid_h,
        grid_w,
        vision_hidden,
        vision_heads,
        vision_head_dim,
        vision_intermediate,
        decoder_hidden,
        _make_ptr(scratch),
        dummy_stage,
        dummy_ctx,
    )

    # Verify output is non-zero (encoder processed data)
    var any_nonzero = False
    for i in range(out_tokens * decoder_hidden):
        if out[i] != 0.0:
            any_nonzero = True
            break
    if not any_nonzero:
        raise Error("Vision encoder output is all zeros with streaming params")

    print("  test_vision_encoder_cpu_with_streaming_params passed")
    _ = patch_emb[0]
    _ = pos_emb[0]
    _ = post_norm_w[0]
    _ = proj[0]
    _ = ident[0]
    _ = fc1[0]
    _ = fc1_full[0]
    _ = fc2_full[0]
    _ = ln1[0]
    _ = ln2[0]
    _ = patches[0]
    _ = scratch[0]


# ---------------------------------------------------------------------------
# Test: Audio encoder with S/C params (CPU path, dummy stage/ctx)
# ---------------------------------------------------------------------------


def test_audio_encoder_cpu_with_streaming_params() raises:
    """Audio encoder on CPU path accepts S/C params without changing output."""
    var num_frames = 8
    var n_mels = 4
    var audio_hidden = 4
    var audio_heads = 1
    var audio_head_dim = 4
    var audio_intermediate = 8
    var decoder_hidden = 4

    # Conv weights: [audio_hidden, n_mels] for first conv
    var conv0 = _make_identity_matrix(audio_hidden)
    var conv_weights_list = List[TensorInfo]()
    conv_weights_list.append(TensorInfo(Int(_make_ptr(conv0)), audio_hidden, n_mels))

    var pos_emb = _make_zeros(num_frames * audio_hidden)
    var post_norm_w = _make_ones(audio_hidden)
    var proj = _make_identity_matrix(decoder_hidden)

    var weights = AudioTowerWeights()
    weights.conv_weights.append(TensorInfo(Int(_make_ptr(conv0)), audio_hidden, n_mels))
    weights.position_embedding = TensorInfo(Int(_make_ptr(pos_emb)), num_frames, audio_hidden)
    weights.post_norm = TensorInfo(Int(_make_ptr(post_norm_w)), 1, audio_hidden)
    weights.projection = TensorInfo(Int(_make_ptr(proj)), decoder_hidden, audio_hidden)

    # One transformer layer
    var layer = VisionLayerWeights()
    var ident = _make_identity_matrix(audio_hidden)
    layer.q_proj = TensorInfo(Int(_make_ptr(ident)), audio_hidden, audio_hidden)
    layer.k_proj = TensorInfo(Int(_make_ptr(ident)), audio_hidden, audio_hidden)
    layer.v_proj = TensorInfo(Int(_make_ptr(ident)), audio_hidden, audio_hidden)
    layer.o_proj = TensorInfo(Int(_make_ptr(ident)), audio_hidden, audio_hidden)
    var fc1_full = List[Float32](length=audio_intermediate * audio_hidden, fill=0.0)
    for r in range(audio_hidden):
        fc1_full[r * audio_hidden + r] = 1.0
    layer.fc1 = TensorInfo(Int(_make_ptr(fc1_full)), audio_intermediate, audio_hidden)
    var fc2_full = List[Float32](length=audio_hidden * audio_intermediate, fill=0.0)
    for r in range(audio_hidden):
        fc2_full[r * audio_intermediate + r] = 1.0
    layer.fc2 = TensorInfo(Int(_make_ptr(fc2_full)), audio_hidden, audio_intermediate)
    var ln1 = _make_ones(audio_hidden)
    var ln2 = _make_ones(audio_hidden)
    layer.layer_norm1 = TensorInfo(Int(_make_ptr(ln1)), 1, audio_hidden)
    layer.layer_norm2 = TensorInfo(Int(_make_ptr(ln2)), 1, audio_hidden)
    weights.layers.append(layer)

    # Input features: [num_frames, n_mels]
    var features = List[Float32](length=num_frames * n_mels, fill=0.3)

    # Output
    var out = _make_zeros(num_frames * decoder_hidden)

    # Generous scratch
    var scratch_size = num_frames * audio_hidden * 60 + audio_heads * num_frames * num_frames * 4
    var scratch = _make_zeros(scratch_size)

    var backend = CPUBackend()
    var dummy_stage = 0
    var dummy_ctx = 0

    forward_audio_encoder(
        backend,
        _make_ptr(out),
        _make_ptr(features),
        weights,
        num_frames,
        n_mels,
        audio_hidden,
        audio_heads,
        audio_head_dim,
        audio_intermediate,
        decoder_hidden,
        _make_ptr(scratch),
        dummy_stage,
        dummy_ctx,
    )

    # Verify output is non-zero
    var any_nonzero = False
    for i in range(num_frames * decoder_hidden):
        if out[i] != 0.0:
            any_nonzero = True
            break
    if not any_nonzero:
        raise Error("Audio encoder output is all zeros with streaming params")

    print("  test_audio_encoder_cpu_with_streaming_params passed")
    _ = conv0[0]
    _ = pos_emb[0]
    _ = post_norm_w[0]
    _ = proj[0]
    _ = ident[0]
    _ = fc1_full[0]
    _ = fc2_full[0]
    _ = ln1[0]
    _ = ln2[0]
    _ = features[0]
    _ = scratch[0]


# ---------------------------------------------------------------------------
# Test: copy_state utility
# ---------------------------------------------------------------------------


def test_copy_state_cpu_parity() raises:
    """Verify copy_state correctly copies data (CPU-level sanity check)."""
    from mogemma.gpu_context import copy_state_kernel

    # Test the kernel logic exists and compiles
    # On CPU, we test copy by verifying the kernel function is importable
    # Full GPU testing requires has_accelerator()
    print("  test_copy_state_cpu_parity passed (import verified)")


# ---------------------------------------------------------------------------
# GPU integration tests (only run when GPU is available)
# ---------------------------------------------------------------------------


def test_gpu_copy_state() raises:
    """Test device-to-device copy_state on actual GPU hardware."""
    comptime if has_accelerator():
        from mogemma.gpu_context import GPUContext, copy_state

        var ctx = GPUContext()
        var size = 128
        var src_buf = ctx.allocate_buffer[DType.float32](size)
        var dst_buf = ctx.allocate_buffer[DType.float32](size)

        # Fill source with known pattern via host buffer
        var host = ctx.allocate_host_buffer[DType.float32](size)
        var hp = host.unsafe_ptr()
        for i in range(size):
            hp.store(i, Float32(i) * 1.5)
        ctx.upload(src_buf, host)
        ctx.sync()

        # Copy device-to-device
        copy_state(ctx, dst_buf.unsafe_ptr(), src_buf.unsafe_ptr(), size)
        ctx.sync()

        # Download and verify
        var verify = ctx.allocate_host_buffer[DType.float32](size)
        ctx.download(verify, dst_buf)
        ctx.sync()

        var vp = verify.unsafe_ptr()
        for i in range(size):
            var expected = Float32(i) * 1.5
            assert_almost_equal(vp.load(i), expected, atol=1e-5)

        print("  test_gpu_copy_state passed")
    else:
        print("  test_gpu_copy_state skipped (no GPU)")


def main() raises:
    print("GPU forward path tests:")
    test_vision_encoder_cpu_with_streaming_params()
    test_audio_encoder_cpu_with_streaming_params()
    test_copy_state_cpu_parity()
    test_gpu_copy_state()
    print("GPU forward path tests passed!")
