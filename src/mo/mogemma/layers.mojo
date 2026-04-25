from std.memory import UnsafePointer
from std.math import sqrt, erf, tanh
from std.sys import has_accelerator
from mogemma.gpu_context import (
    WeightStage,
    GPUContext,
    upload_layer_weights,
    upload_moe_attention_weights,
    upload_vision_layer_weights,
    has_usable_gpu,
)
from mogemma.model import (
    LayerWeights,
    ModelWeights,
    PLELayerWeights,
    MoELayerWeights,
    MoEModelWeights,
    VisionLayerWeights,
    VisionModelWeights,
    AudioTowerWeights,
    TensorInfo,
    KVCache,
    KVCacheTrait,
    PersistentBuffers,
    RoPETables,
    LAYER_TYPE_SLIDING,
    LAYER_TYPE_FULL,
)
from mogemma.ops import (
    vec_mat_mul,
    rope_rotate,
    softmax,
    rms_norm,
    geglu,
    gelu,
    mat_mat_mul,
    mat_mat_mul_i8,
    average_pool_2d,
    top_k,
    ComputeBackend,
    CPUBackend,
)


@always_inline
def _gemm_dispatch[
    B: ComputeBackend
](
    mut backend: B,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    w: TensorInfo,
    batch_size: Int,
    in_dim: Int,
    out_dim: Int,
):
    if w.is_quantized:
        backend.vec_mat_mul_i8(out_ptr, x_ptr, w.i8_ptr, w.scale_ptr, in_dim, out_dim)
    else:
        if batch_size == 1:
            backend.vec_mat_mul(out_ptr, x_ptr, w.ptr, in_dim, out_dim)
        else:
            backend.mat_mat_mul(out_ptr, x_ptr, w.ptr, batch_size, in_dim, out_dim)


@always_inline
def forward_sliding_attention[
    B: ComputeBackend, K: KVCacheTrait
](
    mut backend: B,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    weights: LayerWeights,
    layer_idx: Int,
    pos: Int,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    kv_cache: K,
    rope_tables: RoPETables,
    k_eq_v: Bool,
    attn_logit_softcapping: Float32,
    scratch_ptr: UnsafePointer[Float32, MutAnyOrigin],
):
    """Sliding-window attention for a single token (batch_size=1).

    Uses ring buffer KV cache, theta=10K RoPE with full head_dim rotation,
    and attends only within the sliding window.
    """
    var q_size = num_heads * head_dim
    var kv_size = num_kv_heads * head_dim
    var q_ptr = scratch_ptr
    var k_ptr = scratch_ptr + q_size
    var v_ptr = scratch_ptr + q_size + kv_size

    # 1. Project Q, K, V
    _gemm_dispatch(backend, q_ptr, x_ptr, weights.q_proj, 1, hidden_size, q_size)
    _gemm_dispatch(backend, k_ptr, x_ptr, weights.k_proj, 1, hidden_size, kv_size)
    if k_eq_v:
        # K=V sharing: use K projection results for V too
        _gemm_dispatch(backend, v_ptr, x_ptr, weights.k_proj, 1, hidden_size, kv_size)
    else:
        _gemm_dispatch(backend, v_ptr, x_ptr, weights.v_proj, 1, hidden_size, kv_size)

    # 1b. Per-head QK norms
    if weights.q_norm.ptr != UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=0):
        for h in range(num_heads):
            backend.rms_norm(
                q_ptr + h * head_dim,
                q_ptr + h * head_dim,
                weights.q_norm.ptr,
                head_dim,
                1e-6,
            )
    if weights.k_norm.ptr != UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=0):
        for h in range(num_kv_heads):
            backend.rms_norm(
                k_ptr + h * head_dim,
                k_ptr + h * head_dim,
                weights.k_norm.ptr,
                head_dim,
                1e-6,
            )

    # 2. Apply RoPE (sliding: theta=10K, full head_dim rotation)
    var sliding_freqs = rope_tables.get_sliding_freqs(pos)
    var cos_ptr = sliding_freqs.first
    var sin_ptr = sliding_freqs.second
    for h in range(num_heads):
        backend.rope_rotate(q_ptr + h * head_dim, cos_ptr, sin_ptr, head_dim)
    for h in range(num_kv_heads):
        backend.rope_rotate(k_ptr + h * head_dim, cos_ptr, sin_ptr, head_dim)

    # 3. Write K, V to ring buffer
    var layer_offset = kv_cache.get_layer_offset(layer_idx)
    var window_size = kv_cache.get_window_size()
    backend.kv_write(
        kv_cache.get_k_ptr(),
        k_ptr,
        kv_size,
        pos,
        window_size,
        layer_offset,
        False,
    )
    backend.kv_write(
        kv_cache.get_v_ptr(),
        v_ptr,
        kv_size,
        pos,
        window_size,
        layer_offset,
        False,
    )

    # 4. Compute attention within sliding window
    var attn_range = kv_cache.get_attention_range(layer_idx, pos)
    var valid_len = attn_range.first
    var layer_k_ptr = kv_cache.get_k_ptr() + layer_offset
    var layer_v_ptr = kv_cache.get_v_ptr() + layer_offset

    var scale = 1.0 / sqrt(Float32(head_dim))
    var attn_out_ptr = scratch_ptr + q_size + kv_size + kv_size
    var scores_ptr = attn_out_ptr + q_size

    backend.attention_scores(
        scores_ptr,
        q_ptr,
        layer_k_ptr,
        num_heads,
        num_kv_heads,
        head_dim,
        valid_len,
        kv_size,
        scale,
    )
    if attn_logit_softcapping > 0.0:
        backend.softcap(scores_ptr, num_heads * valid_len, attn_logit_softcapping)

    for h in range(num_heads):
        backend.softmax(scores_ptr + h * valid_len, valid_len)

    backend.attention_value_accum(
        attn_out_ptr,
        scores_ptr,
        layer_v_ptr,
        num_heads,
        num_kv_heads,
        head_dim,
        valid_len,
        kv_size,
    )

    # 5. Output projection
    _gemm_dispatch(backend, out_ptr, attn_out_ptr, weights.o_proj, 1, q_size, hidden_size)


@always_inline
def forward_full_attention[
    B: ComputeBackend, K: KVCacheTrait
](
    mut backend: B,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    weights: LayerWeights,
    layer_idx: Int,
    pos: Int,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    kv_cache: K,
    rope_tables: RoPETables,
    k_eq_v: Bool,
    max_seq_len: Int,  # for scores buffer sizing
    attn_logit_softcapping: Float32,
    scratch_ptr: UnsafePointer[Float32, MutAnyOrigin],
):
    """Full global attention for a single token (batch_size=1).

    Uses linear KV cache, theta=1M RoPE with partial head_dim rotation,
    and attends to all past positions (standard causal).
    """
    var q_size = num_heads * head_dim
    var kv_size = num_kv_heads * head_dim
    var q_ptr = scratch_ptr
    var k_ptr = scratch_ptr + q_size
    var v_ptr = scratch_ptr + q_size + kv_size

    # 1. Project Q, K, V
    _gemm_dispatch(backend, q_ptr, x_ptr, weights.q_proj, 1, hidden_size, q_size)
    _gemm_dispatch(backend, k_ptr, x_ptr, weights.k_proj, 1, hidden_size, kv_size)
    if k_eq_v:
        _gemm_dispatch(backend, v_ptr, x_ptr, weights.k_proj, 1, hidden_size, kv_size)
    else:
        _gemm_dispatch(backend, v_ptr, x_ptr, weights.v_proj, 1, hidden_size, kv_size)

    # 1b. Per-head QK norms
    if weights.q_norm.ptr != UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=0):
        for h in range(num_heads):
            backend.rms_norm(
                q_ptr + h * head_dim,
                q_ptr + h * head_dim,
                weights.q_norm.ptr,
                head_dim,
                1e-6,
            )
    if weights.k_norm.ptr != UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=0):
        for h in range(num_kv_heads):
            backend.rms_norm(
                k_ptr + h * head_dim,
                k_ptr + h * head_dim,
                weights.k_norm.ptr,
                head_dim,
                1e-6,
            )

    # 2. Apply RoPE (full: theta=1M, partial rotation on first rotary_dim dims)
    var rotary_dim = rope_tables.rotary_dim
    var full_freqs = rope_tables.get_full_freqs(pos)
    var cos_ptr = full_freqs.first
    var sin_ptr = full_freqs.second
    for h in range(num_heads):
        # Only rotate first rotary_dim dimensions of each head
        backend.rope_rotate(q_ptr + h * head_dim, cos_ptr, sin_ptr, rotary_dim)
    for h in range(num_kv_heads):
        backend.rope_rotate(k_ptr + h * head_dim, cos_ptr, sin_ptr, rotary_dim)

    # 3. Write K, V to linear cache
    var layer_offset = kv_cache.get_layer_offset(layer_idx)
    var cache_size = kv_cache.get_layer_cache_size(layer_idx)
    backend.kv_write(
        kv_cache.get_k_ptr(),
        k_ptr,
        kv_size,
        pos,
        cache_size,
        layer_offset,
        True,
    )
    backend.kv_write(
        kv_cache.get_v_ptr(),
        v_ptr,
        kv_size,
        pos,
        cache_size,
        layer_offset,
        True,
    )

    # 4. Standard causal attention over all past positions
    var valid_len = pos + 1
    var layer_k_ptr = kv_cache.get_k_ptr() + layer_offset
    var layer_v_ptr = kv_cache.get_v_ptr() + layer_offset

    var scale = 1.0 / sqrt(Float32(head_dim))
    var attn_out_ptr = scratch_ptr + q_size + kv_size + kv_size
    var scores_ptr = attn_out_ptr + q_size

    backend.attention_scores(
        scores_ptr,
        q_ptr,
        layer_k_ptr,
        num_heads,
        num_kv_heads,
        head_dim,
        valid_len,
        kv_size,
        scale,
    )
    if attn_logit_softcapping > 0.0:
        backend.softcap(scores_ptr, num_heads * valid_len, attn_logit_softcapping)

    for h in range(num_heads):
        backend.softmax(scores_ptr + h * valid_len, valid_len)

    backend.attention_value_accum(
        attn_out_ptr,
        scores_ptr,
        layer_v_ptr,
        num_heads,
        num_kv_heads,
        head_dim,
        valid_len,
        kv_size,
    )

    # 5. Output projection
    _gemm_dispatch(backend, out_ptr, attn_out_ptr, weights.o_proj, 1, q_size, hidden_size)


@always_inline
def forward_vision_attention[
    B: ComputeBackend
](
    mut backend: B,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [num_tokens, hidden_size]
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [num_tokens, hidden_size]
    weights: VisionLayerWeights,
    num_tokens: Int,
    hidden_size: Int,
    num_heads: Int,
    head_dim: Int,
    scratch_ptr: UnsafePointer[Float32, MutAnyOrigin],
):
    """Bidirectional multi-head attention for vision transformer.

    All tokens attend to all tokens — no causal mask, no KV cache, no RoPE.
    """
    var q_size = num_tokens * num_heads * head_dim
    var kv_size = num_tokens * num_heads * head_dim
    var q_ptr = scratch_ptr
    var k_ptr = scratch_ptr + q_size
    var v_ptr = scratch_ptr + q_size + kv_size

    # Project Q, K, V: [num_tokens, hidden_size] @ [hidden_size, hidden_size]^T
    _gemm_dispatch(
        backend,
        q_ptr,
        x_ptr,
        weights.q_proj,
        num_tokens,
        hidden_size,
        num_heads * head_dim,
    )
    _gemm_dispatch(
        backend,
        k_ptr,
        x_ptr,
        weights.k_proj,
        num_tokens,
        hidden_size,
        num_heads * head_dim,
    )
    _gemm_dispatch(
        backend,
        v_ptr,
        x_ptr,
        weights.v_proj,
        num_tokens,
        hidden_size,
        num_heads * head_dim,
    )

    var scale = 1.0 / sqrt(Float32(head_dim))
    var attn_out_ptr = scratch_ptr + q_size + kv_size + kv_size
    var scores_ptr = attn_out_ptr + q_size  # [num_heads, num_tokens, num_tokens]

    # Per-head attention
    # On GPU, we should ideally launch a specialized bidirectional attention kernel.
    # For now, we use the same nested loop but dispatch softmax.
    for h in range(num_heads):
        for qi in range(num_tokens):
            var q_head = q_ptr + qi * num_heads * head_dim + h * head_dim
            var s_row = scores_ptr + h * num_tokens * num_tokens + qi * num_tokens

            # Compute scores against all keys
            for ki in range(num_tokens):
                var k_head = k_ptr + ki * num_heads * head_dim + h * head_dim
                var dot: Float32 = 0.0
                for d in range(head_dim):
                    dot += q_head.load(d) * k_head.load(d)
                s_row.store(ki, dot * scale)

            # Softmax over all tokens (bidirectional — no masking)
            backend.softmax(s_row, num_tokens)

            # Weighted sum of values
            var out_head = attn_out_ptr + qi * num_heads * head_dim + h * head_dim
            for d in range(head_dim):
                out_head.store(d, 0.0)
            for vi in range(num_tokens):
                var v_head = v_ptr + vi * num_heads * head_dim + h * head_dim
                var prob = s_row.load(vi)
                for d in range(head_dim):
                    out_head.store(d, out_head.load(d) + prob * v_head.load(d))

    # Output projection
    _gemm_dispatch(
        backend,
        out_ptr,
        attn_out_ptr,
        weights.o_proj,
        num_tokens,
        num_heads * head_dim,
        hidden_size,
    )


@always_inline
def forward_vision_layer[
    B: ComputeBackend
](
    mut backend: B,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [num_tokens, hidden_size]
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [num_tokens, hidden_size]
    weights: VisionLayerWeights,
    num_tokens: Int,
    hidden_size: Int,
    num_heads: Int,
    head_dim: Int,
    intermediate_size: Int,
    scratch_ptr: UnsafePointer[Float32, MutAnyOrigin],
):
    """Single vision transformer layer: LayerNorm + bidirectional attn + residual + LayerNorm + GELU MLP + residual."""
    var total = num_tokens * hidden_size

    # Pre-attention LayerNorm (reuse rms_norm — SigLIP weights are scale-only)
    var norm1_ptr = scratch_ptr
    for t in range(num_tokens):
        backend.rms_norm(
            norm1_ptr + t * hidden_size,
            x_ptr + t * hidden_size,
            weights.layer_norm1.ptr,
            hidden_size,
            1e-6,
        )

    # Bidirectional attention
    var attn_out_ptr = scratch_ptr + total
    var attn_scratch = scratch_ptr + total * 2
    forward_vision_attention(
        backend,
        attn_out_ptr,
        norm1_ptr,
        weights,
        num_tokens,
        hidden_size,
        num_heads,
        head_dim,
        attn_scratch,
    )

    # Attention residual
    var residual_ptr = scratch_ptr + total * 2
    for i in range(total):
        residual_ptr.store(i, x_ptr.load(i) + attn_out_ptr.load(i))

    # Pre-MLP LayerNorm
    var norm2_ptr = scratch_ptr + total * 3
    for t in range(num_tokens):
        backend.rms_norm(
            norm2_ptr + t * hidden_size,
            residual_ptr + t * hidden_size,
            weights.layer_norm2.ptr,
            hidden_size,
            1e-6,
        )

    # Vision MLP: GEGLU — gate = fc1(x), up = fc1_up(x), hidden = gelu(gate) * up, out = fc2(hidden)
    # Gemma 4 vision ships GEGLU-shaped weights (gating_einsum with a 2-axis
    # for gate+up) — see .agents/specs/orbax-safetensors-conversion/e2b-inventory.txt.
    var inter_total = num_tokens * intermediate_size
    var fc1_out_ptr = scratch_ptr + total * 4
    _gemm_dispatch(
        backend,
        fc1_out_ptr,
        norm2_ptr,
        weights.fc1,
        num_tokens,
        hidden_size,
        intermediate_size,
    )

    var fc1_up_out_ptr = scratch_ptr + total * 4 + inter_total
    _gemm_dispatch(
        backend,
        fc1_up_out_ptr,
        norm2_ptr,
        weights.fc1_up,
        num_tokens,
        hidden_size,
        intermediate_size,
    )

    var gelu_out_ptr = scratch_ptr + total * 4 + inter_total * 2
    for t in range(num_tokens):
        var gate_off = t * intermediate_size
        backend.gelu(gelu_out_ptr + gate_off, fc1_out_ptr + gate_off, intermediate_size)
        for d in range(intermediate_size):
            gelu_out_ptr.store(
                gate_off + d,
                gelu_out_ptr.load(gate_off + d) * fc1_up_out_ptr.load(gate_off + d),
            )

    var mlp_out_ptr = scratch_ptr + total * 4 + inter_total * 3
    _gemm_dispatch(
        backend,
        mlp_out_ptr,
        gelu_out_ptr,
        weights.fc2,
        num_tokens,
        intermediate_size,
        hidden_size,
    )

    # MLP residual
    for i in range(total):
        out_ptr.store(i, residual_ptr.load(i) + mlp_out_ptr.load(i))


@always_inline
def forward_vision_encoder[
    B: ComputeBackend, S: AnyType, C: AnyType
](
    mut backend: B,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    patches_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [num_patches, patch_dim]
    weights: VisionModelWeights,
    num_patches: Int,
    grid_h: Int,
    grid_w: Int,
    vision_hidden_size: Int,
    vision_num_heads: Int,
    vision_head_dim: Int,
    vision_intermediate_size: Int,
    decoder_hidden_size: Int,
    scratch_ptr: UnsafePointer[Float32, MutAnyOrigin],
    mut stage: S,
    mut ctx: C,
):
    """Full SigLIP vision encoder: patch embed → position embed → N layers → post-norm → avg pool → project.

    On GPU, layer weights are streamed per-layer via WeightStage. Patch embedding,
    position embedding, post_norm, and projection are used directly (small enough
    or handled by caller).
    """
    var patch_dim = weights.patch_embedding.shape_1
    var total = num_patches * vision_hidden_size

    # 1. Patch embedding: [num_patches, patch_dim] @ [vision_hidden, patch_dim]^T → [num_patches, vision_hidden]
    var embedded_ptr = scratch_ptr
    _gemm_dispatch(
        backend,
        embedded_ptr,
        patches_ptr,
        weights.patch_embedding,
        num_patches,
        patch_dim,
        vision_hidden_size,
    )

    # 2. Add position embeddings (learned, [max_patches, vision_hidden])
    for i in range(total):
        embedded_ptr.store(i, embedded_ptr.load(i) + weights.position_embedding.ptr.load(i))

    # 3. Vision transformer layers
    var current_ptr = embedded_ptr
    var next_ptr = scratch_ptr + total
    var layer_scratch = scratch_ptr + total * 2

    var num_layers = len(weights.layers)
    for l in range(num_layers):
        # Upload layer weights to GPU (when GPU backend)
        var layer_weights = weights.layers[l]
        comptime if has_usable_gpu():
            var stage_ptr = UnsafePointer(to=stage)
            var ctx_ptr = UnsafePointer(to=ctx)
            var stage_ref = rebind[UnsafePointer[WeightStage, MutAnyOrigin]](stage_ptr)
            var ctx_ref = rebind[UnsafePointer[GPUContext, MutAnyOrigin]](ctx_ptr)
            layer_weights = upload_vision_layer_weights(stage_ref[], ctx_ref[], layer_weights)
            ctx_ref[].sync()

        forward_vision_layer(
            backend,
            next_ptr,
            current_ptr,
            layer_weights,
            num_patches,
            vision_hidden_size,
            vision_num_heads,
            vision_head_dim,
            vision_intermediate_size,
            layer_scratch,
        )
        # Swap
        backend.copy(current_ptr, next_ptr, total)

    # 4. Post-LayerNorm
    var norm_ptr = next_ptr
    for t in range(num_patches):
        backend.rms_norm(
            norm_ptr + t * vision_hidden_size,
            current_ptr + t * vision_hidden_size,
            weights.post_norm.ptr,
            vision_hidden_size,
            1e-6,
        )

    # 5. Average pooling (3×3)
    var pool_kernel = 3
    var pooled_ptr = scratch_ptr + total * 2
    backend.average_pool_2d(pooled_ptr, norm_ptr, grid_h, grid_w, vision_hidden_size, pool_kernel)

    # 6. Vision projection → decoder hidden dim
    var pooled_h = grid_h // pool_kernel
    var pooled_w = grid_w // pool_kernel
    var pooled_tokens = pooled_h * pooled_w
    _gemm_dispatch(
        backend,
        out_ptr,
        pooled_ptr,
        weights.projection,
        pooled_tokens,
        vision_hidden_size,
        decoder_hidden_size,
    )


@always_inline
def forward_audio_encoder[
    B: ComputeBackend, S: AnyType, C: AnyType
](
    mut backend: B,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    features_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [n_mels, num_frames] (flattened)
    weights: AudioTowerWeights,
    num_frames: Int,
    n_mels: Int,
    audio_hidden_size: Int,
    audio_num_heads: Int,
    audio_head_dim: Int,
    audio_intermediate_size: Int,
    decoder_hidden_size: Int,
    scratch_ptr: UnsafePointer[Float32, MutAnyOrigin],
    mut stage: S,
    mut ctx: C,
):
    """Audio encoder: conv feature extraction → position embed → transformer layers → post-norm → projection.

    Reuses forward_vision_layer for the transformer layers (same bidirectional attention + GELU MLP).
    Conv layers downsample mel frames into audio_hidden_size-dim tokens.
    On GPU, transformer layer weights are streamed per-layer via WeightStage.
    """
    var num_conv = len(weights.conv_weights)
    var num_tokens = num_frames

    # 1. Conv feature extraction (simplified: treat as linear projections over frames)
    # First conv: [n_mels] → [audio_hidden_size] per frame
    var conv_out_ptr = scratch_ptr
    if num_conv > 0:
        _gemm_dispatch(
            backend,
            conv_out_ptr,
            features_ptr,
            weights.conv_weights[0],
            num_tokens,
            n_mels,
            audio_hidden_size,
        )
        # Subsequent conv layers: [audio_hidden_size] → [audio_hidden_size] with stride-2 downsampling
        for c in range(1, num_conv):
            var new_tokens = num_tokens // 2
            if new_tokens == 0:
                new_tokens = 1
            var next_conv_ptr = conv_out_ptr + num_tokens * audio_hidden_size
            # Stride-2: take every other token, project
            for t in range(new_tokens):
                var src_idx = t * 2
                backend.vec_mat_mul(
                    next_conv_ptr + t * audio_hidden_size,
                    conv_out_ptr + src_idx * audio_hidden_size,
                    weights.conv_weights[c].ptr,
                    audio_hidden_size,
                    audio_hidden_size,
                )
            conv_out_ptr = next_conv_ptr
            num_tokens = new_tokens
    else:
        # No conv: project mel directly
        _gemm_dispatch(
            backend,
            conv_out_ptr,
            features_ptr,
            weights.projection,
            num_tokens,
            n_mels,
            audio_hidden_size,
        )

    var total = num_tokens * audio_hidden_size

    # 2. Add position embeddings
    for i in range(total):
        conv_out_ptr.store(i, conv_out_ptr.load(i) + weights.position_embedding.ptr.load(i))

    # 3. Transformer layers (reuse forward_vision_layer)
    var current_ptr = conv_out_ptr
    var next_ptr = scratch_ptr + total * 3
    var layer_scratch = scratch_ptr + total * 4

    var num_layers = len(weights.layers)
    for l in range(num_layers):
        # Upload layer weights to GPU (when GPU backend)
        var layer_weights = weights.layers[l]
        comptime if has_usable_gpu():
            var stage_ptr = UnsafePointer(to=stage)
            var ctx_ptr = UnsafePointer(to=ctx)
            var stage_ref = rebind[UnsafePointer[WeightStage, MutAnyOrigin]](stage_ptr)
            var ctx_ref = rebind[UnsafePointer[GPUContext, MutAnyOrigin]](ctx_ptr)
            layer_weights = upload_vision_layer_weights(stage_ref[], ctx_ref[], layer_weights)
            ctx_ref[].sync()

        forward_vision_layer(
            backend,
            next_ptr,
            current_ptr,
            layer_weights,
            num_tokens,
            audio_hidden_size,
            audio_num_heads,
            audio_head_dim,
            audio_intermediate_size,
            layer_scratch,
        )
        backend.copy(current_ptr, next_ptr, total)

    # 4. Post-LayerNorm
    var norm_ptr = next_ptr
    for t in range(num_tokens):
        backend.rms_norm(
            norm_ptr + t * audio_hidden_size,
            current_ptr + t * audio_hidden_size,
            weights.post_norm.ptr,
            audio_hidden_size,
            1e-6,
        )

    # 5. Projection → decoder hidden dim
    _gemm_dispatch(
        backend,
        out_ptr,
        norm_ptr,
        weights.projection,
        num_tokens,
        audio_hidden_size,
        decoder_hidden_size,
    )


@always_inline
def forward_mlp[
    B: ComputeBackend
](
    mut backend: B,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [batch_size, hidden_size]
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [batch_size, hidden_size]
    weights: LayerWeights,
    hidden_size: Int,
    intermediate_size: Int,
    scratch_ptr: UnsafePointer[Float32, MutAnyOrigin],  # temp memory
    batch_size: Int = 1,
):
    """Computes the feed-forward network (MLP) block for a standard transformer layer.

    Projects the input hidden state to an intermediate size through gate and up projections, applies the GEGLU activation function, and then down-projects back to the hidden size.
    """
    var gate_ptr = scratch_ptr
    var up_ptr = scratch_ptr + batch_size * intermediate_size
    var geglu_out_ptr = scratch_ptr + batch_size * intermediate_size * 2

    _gemm_dispatch(
        backend,
        gate_ptr,
        x_ptr,
        weights.gate_proj,
        batch_size,
        hidden_size,
        intermediate_size,
    )
    _gemm_dispatch(
        backend,
        up_ptr,
        x_ptr,
        weights.up_proj,
        batch_size,
        hidden_size,
        intermediate_size,
    )

    for b in range(batch_size):
        backend.geglu(
            geglu_out_ptr + b * intermediate_size,
            gate_ptr + b * intermediate_size,
            up_ptr + b * intermediate_size,
            intermediate_size,
        )

    _gemm_dispatch(
        backend,
        out_ptr,
        geglu_out_ptr,
        weights.down_proj,
        batch_size,
        intermediate_size,
        hidden_size,
    )


@always_inline
def forward_gemma4_layer[
    B: ComputeBackend, K: KVCacheTrait
](
    mut backend: B,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [hidden_size]
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [hidden_size]
    weights: LayerWeights,
    layer_idx: Int,
    pos: Int,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    intermediate_size: Int,
    kv_cache: K,
    rope_tables: RoPETables,
    k_eq_v: Bool,
    max_seq_len: Int,
    attn_logit_softcapping: Float32,
    scratch_ptr: UnsafePointer[Float32, MutAnyOrigin],
):
    """Executes a single Gemma 4 transformer layer with attention type dispatch.

    Applies pre-attention RMSNorm, dispatches to sliding or full attention based on layer type,
    adds the attention residual, applies pre-MLP RMSNorm, runs the MLP, and adds the MLP residual.
    """
    # Pre-attention norm
    var norm_x_ptr = scratch_ptr
    backend.rms_norm(norm_x_ptr, x_ptr, weights.input_layernorm.ptr, hidden_size, 1e-6)

    # Attention dispatch
    var attn_out_ptr = scratch_ptr + hidden_size
    var attn_scratch_ptr = scratch_ptr + hidden_size * 2
    if kv_cache.get_layer_type(layer_idx) == LAYER_TYPE_SLIDING:
        forward_sliding_attention(
            backend,
            attn_out_ptr,
            norm_x_ptr,
            weights,
            layer_idx,
            pos,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_cache,
            rope_tables,
            k_eq_v,
            attn_logit_softcapping,
            attn_scratch_ptr,
        )
    else:
        forward_full_attention(
            backend,
            attn_out_ptr,
            norm_x_ptr,
            weights,
            layer_idx,
            pos,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_cache,
            rope_tables,
            k_eq_v,
            max_seq_len,
            attn_logit_softcapping,
            attn_scratch_ptr,
        )

    # Post-attention norm
    var post_attn_ptr = scratch_ptr + hidden_size * 2
    backend.rms_norm(
        post_attn_ptr,
        attn_out_ptr,
        weights.post_attention_layernorm.ptr,
        hidden_size,
        1e-6,
    )

    # Attention residual
    var residual_ptr = scratch_ptr + hidden_size * 3
    backend.vector_add(residual_ptr, x_ptr, post_attn_ptr, hidden_size)

    # Pre-MLP norm
    var norm_residual_ptr = scratch_ptr + hidden_size * 4
    backend.rms_norm(
        norm_residual_ptr,
        residual_ptr,
        weights.pre_feedforward_layernorm.ptr,
        hidden_size,
        1e-6,
    )

    # MLP
    var mlp_out_ptr = scratch_ptr + hidden_size * 5
    var mlp_scratch_ptr = scratch_ptr + hidden_size * 6
    forward_mlp(
        backend,
        mlp_out_ptr,
        norm_residual_ptr,
        weights,
        hidden_size,
        intermediate_size,
        mlp_scratch_ptr,
    )

    # Post-MLP norm
    var post_mlp_ptr = scratch_ptr + hidden_size * 7
    backend.rms_norm(
        post_mlp_ptr,
        mlp_out_ptr,
        weights.post_feedforward_layernorm.ptr,
        hidden_size,
        1e-6,
    )

    # MLP residual
    backend.vector_add(out_ptr, residual_ptr, post_mlp_ptr, hidden_size)


@always_inline
def forward_gemma4_step[
    B: ComputeBackend, K: KVCacheTrait, S: AnyType, C: AnyType, P: AnyType
](
    mut backend: B,
    out_logits_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [vocab_size]
    token_id: Int,
    pos: Int,
    model: ModelWeights,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    intermediate_size: Int,
    vocab_size: Int,
    kv_cache: K,
    rope_tables: RoPETables,
    k_eq_v: Bool,
    max_seq_len: Int,
    attn_logit_softcapping: Float32,
    final_logit_softcapping: Float32,
    scratch_ptr: UnsafePointer[Float32, MutAnyOrigin],
    mut stage: S,
    mut ctx: C,
    persistent: P,
):
    """Executes a single autoregressive generation step for Gemma 4.

    Embeds the token, passes through all transformer layers with hybrid attention dispatch,
    applies final RMSNorm, and projects to logits. Supports GPU weight streaming.
    """
    var num_layers = len(model.layers)
    var current_state = scratch_ptr
    var next_state = scratch_ptr + hidden_size
    var layer_scratch = scratch_ptr + hidden_size * 2

    # Embed and scale
    var emb_scale = sqrt(Float32(hidden_size))
    var embed_ptr = rebind[UnsafePointer[Float32, MutAnyOrigin]](model.embed_tokens.ptr)
    comptime if has_usable_gpu():
        var p_cast = rebind[PersistentBuffers](persistent)
        if p_cast.embed_ptr != UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=0):
            embed_ptr = p_cast.embed_ptr

    backend.embed_lookup(current_state, embed_ptr, token_id, hidden_size, emb_scale)

    # Layer loop with attention type dispatch
    for l in range(num_layers):
        # 1. Orchestrate weights (CPU: use direct, GPU: stream)
        var weights = model.layers[l]
        comptime if has_usable_gpu():
            var stage_ptr = UnsafePointer(to=stage)
            var ctx_ptr = UnsafePointer(to=ctx)
            var stage_ref = rebind[UnsafePointer[WeightStage, MutAnyOrigin]](stage_ptr)
            var ctx_ref = rebind[UnsafePointer[GPUContext, MutAnyOrigin]](ctx_ptr)
            weights = upload_layer_weights(stage_ref[], ctx_ref[], weights)
            ctx_ref[].sync()

        forward_gemma4_layer(
            backend,
            next_state,
            current_state,
            weights,
            l,
            pos,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            kv_cache,
            rope_tables,
            k_eq_v,
            max_seq_len,
            attn_logit_softcapping,
            layer_scratch,
        )
        # Swap states
        backend.copy(current_state, next_state, hidden_size)

    # Final norm + LM head
    var norm_out = next_state
    var norm_ptr = rebind[UnsafePointer[Float32, MutAnyOrigin]](model.norm.ptr)
    var lm_head_ptr = rebind[UnsafePointer[Float32, MutAnyOrigin]](model.lm_head.ptr)
    comptime if has_usable_gpu():
        var p_cast = rebind[PersistentBuffers](persistent)
        if p_cast.norm_ptr != UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=0):
            norm_ptr = p_cast.norm_ptr
        if p_cast.lm_head_ptr != UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=0):
            lm_head_ptr = p_cast.lm_head_ptr

    backend.rms_norm(norm_out, current_state, norm_ptr, hidden_size, 1e-6)
    backend.vec_mat_mul(out_logits_ptr, norm_out, lm_head_ptr, hidden_size, vocab_size)
    if final_logit_softcapping > 0.0:
        backend.softcap(out_logits_ptr, vocab_size, final_logit_softcapping)


@always_inline
def forward_gemma4_step_with_embedding[
    B: ComputeBackend, K: KVCacheTrait, S: AnyType, C: AnyType, P: AnyType
](
    mut backend: B,
    out_logits_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [vocab_size]
    embedding_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [hidden_size] — pre-computed embedding
    pos: Int,
    model: ModelWeights,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    intermediate_size: Int,
    vocab_size: Int,
    kv_cache: K,
    rope_tables: RoPETables,
    k_eq_v: Bool,
    max_seq_len: Int,
    attn_logit_softcapping: Float32,
    final_logit_softcapping: Float32,
    scratch_ptr: UnsafePointer[Float32, MutAnyOrigin],
    mut stage: S,
    mut ctx: C,
    persistent: P,
):
    """Like forward_gemma4_step but uses a pre-computed embedding instead of token lookup.

    Used for vision token injection during prefill.
    """
    var num_layers = len(model.layers)
    var current_state = scratch_ptr
    var next_state = scratch_ptr + hidden_size
    var layer_scratch = scratch_ptr + hidden_size * 2

    # Use provided embedding directly (already scaled by caller)
    backend.copy(current_state, embedding_ptr, hidden_size)

    # Layer loop (same as forward_gemma4_step)
    for l in range(num_layers):
        var weights = model.layers[l]
        comptime if has_usable_gpu():
            var stage_ptr = UnsafePointer(to=stage)
            var ctx_ptr = UnsafePointer(to=ctx)
            var stage_ref = rebind[UnsafePointer[WeightStage, MutAnyOrigin]](stage_ptr)
            var ctx_ref = rebind[UnsafePointer[GPUContext, MutAnyOrigin]](ctx_ptr)
            weights = upload_layer_weights(stage_ref[], ctx_ref[], weights)
            ctx_ref[].sync()

        forward_gemma4_layer(
            backend,
            next_state,
            current_state,
            weights,
            l,
            pos,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            kv_cache,
            rope_tables,
            k_eq_v,
            max_seq_len,
            attn_logit_softcapping,
            layer_scratch,
        )
        backend.copy(current_state, next_state, hidden_size)

    # Final norm + LM head — use persistent GPU buffers when available
    var norm_out = next_state
    var norm_ptr = rebind[UnsafePointer[Float32, MutAnyOrigin]](model.norm.ptr)
    var lm_head_ptr = rebind[UnsafePointer[Float32, MutAnyOrigin]](model.lm_head.ptr)
    comptime if has_usable_gpu():
        var p_cast = rebind[PersistentBuffers](persistent)
        if p_cast.norm_ptr != UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=0):
            norm_ptr = p_cast.norm_ptr
        if p_cast.lm_head_ptr != UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=0):
            lm_head_ptr = p_cast.lm_head_ptr

    backend.rms_norm(norm_out, current_state, norm_ptr, hidden_size, 1e-6)
    backend.vec_mat_mul(out_logits_ptr, norm_out, lm_head_ptr, hidden_size, vocab_size)
    if final_logit_softcapping > 0.0:
        backend.softcap(out_logits_ptr, vocab_size, final_logit_softcapping)


# ── PLE (Per-Layer Embedding) for E2B/E4B ────────────────────────────────


@always_inline
def forward_ple_input[
    B: ComputeBackend
](
    mut backend: B,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    token_id: Int,
    ple_weights: PLELayerWeights,
    hidden_size: Int,
    ple_dim: Int,
    scratch_ptr: UnsafePointer[Float32, MutAnyOrigin],
):
    """Inject per-layer embedding into hidden state: embed → project → norm → add."""
    var embed_ptr = scratch_ptr
    var proj_ptr = scratch_ptr + ple_dim
    var emb_src = ple_weights.per_layer_embedding.ptr + token_id * ple_dim
    for i in range(ple_dim):
        embed_ptr.store(i, emb_src.load(i))
    backend.vec_mat_mul(
        proj_ptr,
        embed_ptr,
        ple_weights.per_layer_projection.ptr,
        ple_dim,
        hidden_size,
    )
    var normed_ptr = scratch_ptr + ple_dim + hidden_size
    backend.rms_norm(normed_ptr, proj_ptr, ple_weights.per_layer_norm.ptr, hidden_size, 1e-6)
    backend.vector_add(out_ptr, out_ptr, normed_ptr, hidden_size)


@always_inline
def forward_gemma4_ple_step[
    B: ComputeBackend, K: KVCacheTrait, S: AnyType, C: AnyType, P: AnyType
](
    mut backend: B,
    out_logits_ptr: UnsafePointer[Float32, MutAnyOrigin],
    token_id: Int,
    pos: Int,
    model: ModelWeights,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    intermediate_size: Int,
    vocab_size: Int,
    ple_dim: Int,
    kv_cache: K,
    rope_tables: RoPETables,
    k_eq_v: Bool,
    max_seq_len: Int,
    attn_logit_softcapping: Float32,
    final_logit_softcapping: Float32,
    kv_sharing_map_ptr: UnsafePointer[Int64, MutExternalOrigin],
    num_kv_sharing_layers: Int,
    scratch_ptr: UnsafePointer[Float32, MutAnyOrigin],
    mut stage: S,
    mut ctx: C,
    persistent: P,
):
    """E2B/E4B forward step with PLE injection and optional shared-KV attention."""
    var num_layers = len(model.layers)
    var current_state = scratch_ptr
    var next_state = scratch_ptr + hidden_size
    var layer_scratch = scratch_ptr + hidden_size * 2
    var emb_scale = sqrt(Float32(hidden_size))
    backend.embed_lookup(current_state, model.embed_tokens.ptr, token_id, hidden_size, emb_scale)
    for l in range(num_layers):
        var weights = model.layers[l]
        var ple_weights = PLELayerWeights()  # placeholder if no PLE
        if model.has_ple and l < len(model.ple_layers):
            ple_weights = model.ple_layers[l]

        comptime if has_usable_gpu():
            var stage_ptr = UnsafePointer(to=stage)
            var ctx_ptr = UnsafePointer(to=ctx)
            var stage_ref = rebind[UnsafePointer[WeightStage, MutAnyOrigin]](stage_ptr)
            var ctx_ref = rebind[UnsafePointer[GPUContext, MutAnyOrigin]](ctx_ptr)
            weights = upload_layer_weights(stage_ref[], ctx_ref[], weights)
            ctx_ref[].sync()

        if model.has_ple and l < len(model.ple_layers):
            forward_ple_input(
                backend,
                current_state,
                token_id,
                ple_weights,
                hidden_size,
                ple_dim,
                layer_scratch,
            )
        var source_layer = -1
        if num_kv_sharing_layers > 0 and l < num_kv_sharing_layers:
            source_layer = Int(kv_sharing_map_ptr.load(l))
        if source_layer >= 0:
            forward_gemma4_layer(
                backend,
                next_state,
                current_state,
                weights,
                source_layer,
                pos,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                intermediate_size,
                kv_cache,
                rope_tables,
                k_eq_v,
                max_seq_len,
                attn_logit_softcapping,
                layer_scratch,
            )
        else:
            forward_gemma4_layer(
                backend,
                next_state,
                current_state,
                weights,
                l,
                pos,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                intermediate_size,
                kv_cache,
                rope_tables,
                k_eq_v,
                max_seq_len,
                attn_logit_softcapping,
                layer_scratch,
            )
        backend.copy(current_state, next_state, hidden_size)
    var norm_out_ple = next_state
    backend.rms_norm(norm_out_ple, current_state, model.norm.ptr, hidden_size, 1e-6)
    backend.vec_mat_mul(out_logits_ptr, norm_out_ple, model.lm_head.ptr, hidden_size, vocab_size)
    if final_logit_softcapping > 0.0:
        backend.softcap(out_logits_ptr, vocab_size, final_logit_softcapping)


# ── MoE (Mixture of Experts) for 26B ─────────────────────────────────────


@always_inline
def forward_moe_router[
    B: ComputeBackend
](
    mut backend: B,
    expert_indices_ptr: UnsafePointer[Int32, MutAnyOrigin],
    expert_weights_ptr: UnsafePointer[Float32, MutAnyOrigin],
    hidden_ptr: UnsafePointer[Float32, MutAnyOrigin],
    router_proj: TensorInfo,
    router_scale: TensorInfo,
    per_expert_scale: TensorInfo,
    hidden_size: Int,
    num_experts: Int,
    k: Int,
    scratch_ptr: UnsafePointer[Float32, MutAnyOrigin],
):
    """Route hidden state to top-k experts using the Gemma 4 MoE contract."""
    var normalized_ptr = scratch_ptr
    var sum_sq: Float32 = 0.0
    for i in range(hidden_size):
        var value = hidden_ptr.load(i)
        sum_sq += value * value
    var inv_rms = 1.0 / sqrt(sum_sq / Float32(hidden_size) + 1e-6)
    var inv_hidden = 1.0 / sqrt(Float32(hidden_size))
    for i in range(hidden_size):
        var value = hidden_ptr.load(i) * inv_rms
        value *= router_scale.ptr.load(i)
        normalized_ptr.store(i, value * inv_hidden)

    var logits_ptr = scratch_ptr + hidden_size
    _gemm_dispatch(
        backend,
        logits_ptr,
        normalized_ptr,
        router_proj,
        1,
        hidden_size,
        num_experts,
    )
    backend.softmax(logits_ptr, num_experts)
    backend.top_k(logits_ptr, k, num_experts, expert_indices_ptr, expert_weights_ptr)

    var weight_sum: Float32 = 0.0
    for i in range(k):
        weight_sum += expert_weights_ptr.load(i)
    if weight_sum > 0.0:
        var inv_sum = 1.0 / weight_sum
        for i in range(k):
            expert_weights_ptr.store(i, expert_weights_ptr.load(i) * inv_sum)

    for i in range(k):
        var expert_idx = Int(expert_indices_ptr.load(i))
        var scaled = expert_weights_ptr.load(i) * per_expert_scale.ptr.load(expert_idx)
        expert_weights_ptr.store(i, scaled)


@always_inline
def forward_moe_experts[
    B: ComputeBackend, S: AnyType, C: AnyType
](
    mut backend: B,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    hidden_ptr: UnsafePointer[Float32, MutAnyOrigin],
    expert_indices_ptr: UnsafePointer[Int32, MutAnyOrigin],
    expert_weights_ptr: UnsafePointer[Float32, MutAnyOrigin],
    expert_gate_up_proj: TensorInfo,
    expert_down_proj: TensorInfo,
    k: Int,
    hidden_size: Int,
    intermediate_size: Int,
    scratch_ptr: UnsafePointer[Float32, MutAnyOrigin],
    mut stage: S,
    mut ctx: C,
):
    """Execute the selected packed experts and compute the weighted sum."""
    var gate_ptr = scratch_ptr
    var up_ptr = scratch_ptr + intermediate_size
    var geglu_out_ptr = scratch_ptr + intermediate_size * 2
    var expert_out_ptr = scratch_ptr + intermediate_size * 3
    var gate_up_stride = 2 * intermediate_size * hidden_size
    var down_stride = hidden_size * intermediate_size

    # Zero the output accumulator
    for i in range(hidden_size):
        out_ptr.store(i, 0.0)

    for sel in range(k):
        var idx = Int(expert_indices_ptr.load(sel))
        var weight = expert_weights_ptr.load(sel)

        var gate_up_base = expert_gate_up_proj.ptr + idx * gate_up_stride
        var gate_tensor = TensorInfo(Int(gate_up_base), intermediate_size, hidden_size)
        var up_tensor = TensorInfo(
            Int(gate_up_base + intermediate_size * hidden_size),
            intermediate_size,
            hidden_size,
        )
        var down_tensor = TensorInfo(
            Int(expert_down_proj.ptr + idx * down_stride),
            hidden_size,
            intermediate_size,
        )

        _gemm_dispatch(
            backend,
            gate_ptr,
            hidden_ptr,
            gate_tensor,
            1,
            hidden_size,
            intermediate_size,
        )
        _gemm_dispatch(
            backend,
            up_ptr,
            hidden_ptr,
            up_tensor,
            1,
            hidden_size,
            intermediate_size,
        )
        backend.geglu(geglu_out_ptr, gate_ptr, up_ptr, intermediate_size)
        _gemm_dispatch(
            backend,
            expert_out_ptr,
            geglu_out_ptr,
            down_tensor,
            1,
            intermediate_size,
            hidden_size,
        )

        backend.vector_add_scaled(out_ptr, out_ptr, expert_out_ptr, weight, hidden_size)

    _ = stage
    _ = ctx


@always_inline
def forward_moe_layer[
    B: ComputeBackend, K: KVCacheTrait, S: AnyType, C: AnyType
](
    mut backend: B,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    weights: MoELayerWeights,
    layer_idx: Int,
    pos: Int,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    num_experts: Int,
    moe_top_k: Int,
    moe_intermediate_size: Int,
    kv_cache: K,
    rope_tables: RoPETables,
    max_seq_len: Int,
    attn_logit_softcapping: Float32,
    scratch_ptr: UnsafePointer[Float32, MutAnyOrigin],
    mut stage: S,
    mut ctx: C,
):
    """Single Gemma 4 MoE layer: attention + dense branch + routed expert branch."""
    var norm_x_ptr = scratch_ptr
    backend.rms_norm(norm_x_ptr, x_ptr, weights.input_layernorm.ptr, hidden_size, 1e-6)
    var q_size = num_heads * head_dim
    var kv_size = num_kv_heads * head_dim
    var attn_out_ptr = scratch_ptr + hidden_size
    var attn_scratch_ptr = scratch_ptr + hidden_size * 2
    var q_ptr = attn_scratch_ptr
    var k_ptr = attn_scratch_ptr + q_size
    var v_ptr = attn_scratch_ptr + q_size + kv_size
    _gemm_dispatch(backend, q_ptr, norm_x_ptr, weights.q_proj, 1, hidden_size, q_size)
    _gemm_dispatch(backend, k_ptr, norm_x_ptr, weights.k_proj, 1, hidden_size, kv_size)
    _gemm_dispatch(backend, v_ptr, norm_x_ptr, weights.v_proj, 1, hidden_size, kv_size)
    if weights.q_norm.ptr != UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=0):
        for h in range(num_heads):
            backend.rms_norm(
                q_ptr + h * head_dim,
                q_ptr + h * head_dim,
                weights.q_norm.ptr,
                head_dim,
                1e-6,
            )
    if weights.k_norm.ptr != UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=0):
        for h in range(num_kv_heads):
            backend.rms_norm(
                k_ptr + h * head_dim,
                k_ptr + h * head_dim,
                weights.k_norm.ptr,
                head_dim,
                1e-6,
            )

    # 2. Apply RoPE (hybrid logic)
    if kv_cache.get_layer_type(layer_idx) == LAYER_TYPE_SLIDING:
        var sliding_freqs = rope_tables.get_sliding_freqs(pos)
        var cos_ptr = sliding_freqs.first
        var sin_ptr = sliding_freqs.second
        for h in range(num_heads):
            backend.rope_rotate(q_ptr + h * head_dim, cos_ptr, sin_ptr, head_dim)
        for h in range(num_kv_heads):
            backend.rope_rotate(k_ptr + h * head_dim, cos_ptr, sin_ptr, head_dim)
    else:
        var rotary_dim = rope_tables.rotary_dim
        var full_freqs = rope_tables.get_full_freqs(pos)
        var cos_ptr = full_freqs.first
        var sin_ptr = full_freqs.second
        for h in range(num_heads):
            backend.rope_rotate(q_ptr + h * head_dim, cos_ptr, sin_ptr, rotary_dim)
        for h in range(num_kv_heads):
            backend.rope_rotate(k_ptr + h * head_dim, cos_ptr, sin_ptr, rotary_dim)

    # 3. Write K, V to cache
    var layer_offset = kv_cache.get_layer_offset(layer_idx)
    var cache_size = kv_cache.get_layer_cache_size(layer_idx)
    var is_full = kv_cache.get_layer_type(layer_idx) == LAYER_TYPE_FULL
    backend.kv_write(
        kv_cache.get_k_ptr(),
        k_ptr,
        kv_size,
        pos,
        cache_size,
        layer_offset,
        is_full,
    )
    backend.kv_write(
        kv_cache.get_v_ptr(),
        v_ptr,
        kv_size,
        pos,
        cache_size,
        layer_offset,
        is_full,
    )

    # 4. Attention
    var attn_range = kv_cache.get_attention_range(layer_idx, pos)
    var valid_len = attn_range.first
    var layer_k_ptr = kv_cache.get_k_ptr() + layer_offset
    var layer_v_ptr = kv_cache.get_v_ptr() + layer_offset
    var scale = 1.0 / sqrt(Float32(head_dim))
    var attn_weighted_ptr = attn_scratch_ptr + q_size + kv_size + kv_size
    var scores_ptr = attn_weighted_ptr + q_size

    backend.attention_scores(
        scores_ptr,
        q_ptr,
        layer_k_ptr,
        num_heads,
        num_kv_heads,
        head_dim,
        valid_len,
        kv_size,
        scale,
    )
    if attn_logit_softcapping > 0.0:
        backend.softcap(scores_ptr, num_heads * valid_len, attn_logit_softcapping)
    for h in range(num_heads):
        backend.softmax(scores_ptr + h * valid_len, valid_len)
    backend.attention_value_accum(
        attn_weighted_ptr,
        scores_ptr,
        layer_v_ptr,
        num_heads,
        num_kv_heads,
        head_dim,
        valid_len,
        kv_size,
    )

    _gemm_dispatch(
        backend,
        attn_out_ptr,
        attn_weighted_ptr,
        weights.o_proj,
        1,
        q_size,
        hidden_size,
    )
    var post_attn_ptr = scratch_ptr + hidden_size * 2
    backend.rms_norm(
        post_attn_ptr,
        attn_out_ptr,
        weights.post_attention_layernorm.ptr,
        hidden_size,
        1e-6,
    )
    var residual_ptr = scratch_ptr + hidden_size * 3
    backend.vector_add(residual_ptr, x_ptr, post_attn_ptr, hidden_size)
    var dense_norm_ptr = scratch_ptr + hidden_size * 4
    backend.rms_norm(
        dense_norm_ptr,
        residual_ptr,
        weights.pre_feedforward_layernorm.ptr,
        hidden_size,
        1e-6,
    )
    var dense_out_ptr = scratch_ptr + hidden_size * 5
    var dense_post_ptr = scratch_ptr + hidden_size * 6
    var moe_norm_ptr = scratch_ptr + hidden_size * 7
    backend.rms_norm(
        moe_norm_ptr,
        residual_ptr,
        weights.pre_feedforward_layernorm_2.ptr,
        hidden_size,
        1e-6,
    )
    var moe_out_ptr = scratch_ptr + hidden_size * 8
    var post_moe_ptr = scratch_ptr + hidden_size * 9
    var combine_ptr = scratch_ptr + hidden_size * 10
    var dense_intermediate_size = weights.dense_gate_proj.shape_0
    var dense_scratch = scratch_ptr + hidden_size * 11
    var moe_scratch = dense_scratch + dense_intermediate_size * 3
    var expert_indices_ptr = UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=Int(moe_scratch))
    var expert_weights_ptr = moe_scratch + moe_top_k
    var router_work_ptr = moe_scratch + moe_top_k * 2
    var expert_scratch = router_work_ptr + hidden_size + num_experts

    var dense_weights = LayerWeights()
    dense_weights.gate_proj = weights.dense_gate_proj
    dense_weights.up_proj = weights.dense_up_proj
    dense_weights.down_proj = weights.dense_down_proj
    forward_mlp(
        backend,
        dense_out_ptr,
        dense_norm_ptr,
        dense_weights,
        hidden_size,
        dense_intermediate_size,
        dense_scratch,
    )
    backend.rms_norm(
        dense_post_ptr,
        dense_out_ptr,
        weights.post_feedforward_layernorm_1.ptr,
        hidden_size,
        1e-6,
    )

    # Router
    # Router logits consume x1 via no-scale RMSNorm; expert execution uses the MoE pre-norm branch.
    forward_moe_router(
        backend,
        expert_indices_ptr,
        expert_weights_ptr,
        residual_ptr,
        weights.router_proj,
        weights.router_scale,
        weights.per_expert_scale,
        hidden_size,
        num_experts,
        moe_top_k,
        router_work_ptr,
    )

    forward_moe_experts(
        backend,
        moe_out_ptr,
        moe_norm_ptr,
        expert_indices_ptr,
        expert_weights_ptr,
        weights.expert_gate_up_proj,
        weights.expert_down_proj,
        moe_top_k,
        hidden_size,
        moe_intermediate_size,
        expert_scratch,
        stage,
        ctx,
    )

    backend.rms_norm(
        post_moe_ptr,
        moe_out_ptr,
        weights.post_feedforward_layernorm_2.ptr,
        hidden_size,
        1e-6,
    )
    backend.vector_add(combine_ptr, dense_post_ptr, post_moe_ptr, hidden_size)
    # H-A (HF-ref confirmed): post_feedforward_layernorm applied to (h1 + h2) before residual add.
    # See .agents/knowledge/gemma4-models.md "MoE layer — post_feedforward_layernorm".
    # Reuse post_moe_ptr as scratch for the normed combined branches.
    if weights.post_feedforward_layernorm.ptr != UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=0):
        backend.rms_norm(
            post_moe_ptr,
            combine_ptr,
            weights.post_feedforward_layernorm.ptr,
            hidden_size,
            1e-6,
        )
        backend.vector_add(out_ptr, residual_ptr, post_moe_ptr, hidden_size)
    else:
        backend.vector_add(out_ptr, residual_ptr, combine_ptr, hidden_size)
    # layer_scalar (HF name) / moe_skip_scale (Orbax name): multiplicative scalar on the entire
    # layer output, applied last. See .agents/knowledge/gemma4-models.md "MoE layer — skip_scale".
    if weights.moe_skip_scale.ptr != UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=0):
        var layer_scalar = weights.moe_skip_scale.ptr[0]
        for i in range(hidden_size):
            out_ptr[i] = out_ptr[i] * layer_scalar


@always_inline
def forward_gemma4_moe_step[
    B: ComputeBackend, K: KVCacheTrait, S: AnyType, C: AnyType, P: AnyType
](
    mut backend: B,
    out_logits_ptr: UnsafePointer[Float32, MutAnyOrigin],
    token_id: Int,
    pos: Int,
    model: MoEModelWeights,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    num_experts: Int,
    moe_top_k: Int,
    moe_intermediate_size: Int,
    vocab_size: Int,
    kv_cache: K,
    rope_tables: RoPETables,
    max_seq_len: Int,
    attn_logit_softcapping: Float32,
    final_logit_softcapping: Float32,
    scratch_ptr: UnsafePointer[Float32, MutAnyOrigin],
    mut stage: S,
    mut ctx: C,
    persistent: P,
):
    """Full 26B MoE forward step: embed → MoE layers → norm → logits."""
    var num_layers = len(model.layers)
    var current_state = scratch_ptr
    var next_state = scratch_ptr + hidden_size
    var layer_scratch = scratch_ptr + hidden_size * 2

    # Embed and scale — use persistent GPU buffer when available
    var emb_scale = sqrt(Float32(hidden_size))
    var embed_ptr = rebind[UnsafePointer[Float32, MutAnyOrigin]](model.embed_tokens.ptr)
    comptime if has_usable_gpu():
        var p_cast = rebind[PersistentBuffers](persistent)
        if p_cast.embed_ptr != UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=0):
            embed_ptr = p_cast.embed_ptr

    backend.embed_lookup(current_state, embed_ptr, token_id, hidden_size, emb_scale)

    # Layer loop with 2-phase weight streaming
    for l in range(num_layers):
        var weights = model.layers[l]
        comptime if has_usable_gpu():
            var stage_ptr = UnsafePointer(to=stage)
            var ctx_ptr = UnsafePointer(to=ctx)
            var stage_ref = rebind[UnsafePointer[WeightStage, MutAnyOrigin]](stage_ptr)
            var ctx_ref = rebind[UnsafePointer[GPUContext, MutAnyOrigin]](ctx_ptr)
            weights = upload_moe_attention_weights(stage_ref[], ctx_ref[], weights)
            ctx_ref[].sync()

        forward_moe_layer(
            backend,
            next_state,
            current_state,
            weights,
            l,
            pos,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            num_experts,
            moe_top_k,
            moe_intermediate_size,
            kv_cache,
            rope_tables,
            max_seq_len,
            attn_logit_softcapping,
            layer_scratch,
            stage,
            ctx,
        )
        backend.copy(current_state, next_state, hidden_size)

    # Final norm + LM head — use persistent GPU buffers when available
    var norm_out_moe = next_state
    var norm_ptr = rebind[UnsafePointer[Float32, MutAnyOrigin]](model.norm.ptr)
    var lm_head_ptr = rebind[UnsafePointer[Float32, MutAnyOrigin]](model.lm_head.ptr)
    comptime if has_usable_gpu():
        var p_cast = rebind[PersistentBuffers](persistent)
        if p_cast.norm_ptr != UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=0):
            norm_ptr = p_cast.norm_ptr
        if p_cast.lm_head_ptr != UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=0):
            lm_head_ptr = p_cast.lm_head_ptr

    backend.rms_norm(norm_out_moe, current_state, norm_ptr, hidden_size, 1e-6)
    backend.vec_mat_mul(out_logits_ptr, norm_out_moe, lm_head_ptr, hidden_size, vocab_size)
    if final_logit_softcapping > 0.0:
        backend.softcap(out_logits_ptr, vocab_size, final_logit_softcapping)
