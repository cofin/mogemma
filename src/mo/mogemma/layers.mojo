from std.memory import UnsafePointer
from std.math import sqrt, erf, tanh
from mogemma.model import (
    LayerWeights,
    ModelWeights,
    PLELayerWeights,
    MoEExpertWeights,
    MoELayerWeights,
    MoEModelWeights,
    VisionLayerWeights,
    VisionModelWeights,
    AudioTowerWeights,
    TensorInfo,
    KVCache,
    RoPETables,
    LAYER_TYPE_SLIDING,
    LAYER_TYPE_FULL,
)
from mogemma.ops import vec_mat_mul, rope_rotate, softmax, rms_norm, geglu, gelu, mat_mat_mul, mat_mat_mul_i8, average_pool_2d, top_k


@always_inline
fn _gemm_dispatch(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],
    w: TensorInfo,
    batch_size: Int,
    in_dim: Int,
    out_dim: Int,
):
    if w.is_quantized:
        mat_mat_mul_i8(out_ptr, x_ptr, w.i8_ptr, w.scale_ptr, batch_size, in_dim, out_dim)
    else:
        mat_mat_mul(out_ptr, x_ptr, w.ptr, batch_size, in_dim, out_dim)


@always_inline
fn forward_sliding_attention(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],  # [hidden_size]
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],    # [hidden_size]
    weights: LayerWeights,
    layer_idx: Int,
    pos: Int,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    kv_cache: KVCache,
    rope_tables: RoPETables,
    k_eq_v: Bool,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin],
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
    _gemm_dispatch(q_ptr, x_ptr, weights.q_proj, 1, hidden_size, q_size)
    _gemm_dispatch(k_ptr, x_ptr, weights.k_proj, 1, hidden_size, kv_size)
    if k_eq_v:
        # K=V sharing: use K projection weights for V too
        _gemm_dispatch(v_ptr, x_ptr, weights.k_proj, 1, hidden_size, kv_size)
    else:
        _gemm_dispatch(v_ptr, x_ptr, weights.v_proj, 1, hidden_size, kv_size)

    # 1b. Per-head QK norms
    if weights.q_norm.ptr != UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=0):
        for h in range(num_heads):
            rms_norm(q_ptr + h * head_dim, q_ptr + h * head_dim, weights.q_norm.ptr, head_dim, 1e-6)
    if weights.k_norm.ptr != UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=0):
        for h in range(num_kv_heads):
            rms_norm(k_ptr + h * head_dim, k_ptr + h * head_dim, weights.k_norm.ptr, head_dim, 1e-6)

    # 2. Apply RoPE (sliding: theta=10K, full head_dim rotation)
    var sliding_freqs = rope_tables.get_sliding_freqs(pos)
    var cos_ptr = sliding_freqs.first
    var sin_ptr = sliding_freqs.second
    for h in range(num_heads):
        rope_rotate(q_ptr + h * head_dim, cos_ptr, sin_ptr, head_dim)
    for h in range(num_kv_heads):
        rope_rotate(k_ptr + h * head_dim, cos_ptr, sin_ptr, head_dim)

    # 3. Write K, V to ring buffer
    kv_cache.write_kv(layer_idx, pos, k_ptr, v_ptr)

    # 4. Compute attention within sliding window
    var attn_range = kv_cache.get_attention_range(layer_idx, pos)
    var valid_len = attn_range.first
    var cache_size = attn_range.second
    var kv_ptrs = kv_cache.get_kv_ptrs(layer_idx)
    var layer_k_ptr = kv_ptrs.first
    var layer_v_ptr = kv_ptrs.second

    var heads_per_kv = num_heads // num_kv_heads
    var scale = 1.0 / sqrt(Float32(head_dim))
    var attn_out_ptr = scratch_ptr + q_size + kv_size + kv_size
    var window_size = kv_cache.window_size

    for h in range(num_heads):
        var kv_h = h // heads_per_kv
        var q_head_ptr = q_ptr + h * head_dim
        var scores_ptr = attn_out_ptr + q_size + h * window_size

        # Score against valid positions in ring buffer
        if pos < window_size:
            # No wrap-around yet — positions 0..pos are sequential
            for t in range(valid_len):
                var k_head_ptr = layer_k_ptr + t * kv_size + kv_h * head_dim
                var score: Float32 = 0.0
                for d in range(head_dim):
                    score += q_head_ptr.load(d) * k_head_ptr.load(d)
                scores_ptr.store(t, score * scale)
        else:
            # Ring buffer has wrapped — iterate over all `window_size` slots
            # Each slot holds the token at `(slot_pos)` where slot_pos is the
            # original sequence position that was written there.
            # We need causal masking: only attend to positions <= pos.
            # Since the buffer is full and all entries are within the window,
            # all `window_size` entries are valid and within causal range.
            for t in range(valid_len):
                var k_head_ptr = layer_k_ptr + t * kv_size + kv_h * head_dim
                var score: Float32 = 0.0
                for d in range(head_dim):
                    score += q_head_ptr.load(d) * k_head_ptr.load(d)
                scores_ptr.store(t, score * scale)

        softmax(scores_ptr, valid_len)

        # Weighted sum of V
        var out_head_ptr = attn_out_ptr + h * head_dim
        for d in range(head_dim):
            out_head_ptr.store(d, 0.0)
        for t in range(valid_len):
            var v_head_ptr = layer_v_ptr + t * kv_size + kv_h * head_dim
            var prob = scores_ptr.load(t)
            for d in range(head_dim):
                out_head_ptr.store(d, out_head_ptr.load(d) + prob * v_head_ptr.load(d))

    # 5. Output projection
    _gemm_dispatch(out_ptr, attn_out_ptr, weights.o_proj, 1, q_size, hidden_size)


@always_inline
fn forward_full_attention(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],  # [hidden_size]
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],    # [hidden_size]
    weights: LayerWeights,
    layer_idx: Int,
    pos: Int,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    kv_cache: KVCache,
    rope_tables: RoPETables,
    k_eq_v: Bool,
    max_seq_len: Int,  # for scores buffer sizing
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin],
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
    _gemm_dispatch(q_ptr, x_ptr, weights.q_proj, 1, hidden_size, q_size)
    _gemm_dispatch(k_ptr, x_ptr, weights.k_proj, 1, hidden_size, kv_size)
    if k_eq_v:
        _gemm_dispatch(v_ptr, x_ptr, weights.k_proj, 1, hidden_size, kv_size)
    else:
        _gemm_dispatch(v_ptr, x_ptr, weights.v_proj, 1, hidden_size, kv_size)

    # 1b. Per-head QK norms
    if weights.q_norm.ptr != UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=0):
        for h in range(num_heads):
            rms_norm(q_ptr + h * head_dim, q_ptr + h * head_dim, weights.q_norm.ptr, head_dim, 1e-6)
    if weights.k_norm.ptr != UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=0):
        for h in range(num_kv_heads):
            rms_norm(k_ptr + h * head_dim, k_ptr + h * head_dim, weights.k_norm.ptr, head_dim, 1e-6)

    # 2. Apply RoPE (full: theta=1M, partial rotation on first rotary_dim dims)
    var rotary_dim = rope_tables.rotary_dim
    var full_freqs = rope_tables.get_full_freqs(pos)
    var cos_ptr = full_freqs.first
    var sin_ptr = full_freqs.second
    for h in range(num_heads):
        # Only rotate first rotary_dim dimensions of each head
        rope_rotate(q_ptr + h * head_dim, cos_ptr, sin_ptr, rotary_dim)
    for h in range(num_kv_heads):
        rope_rotate(k_ptr + h * head_dim, cos_ptr, sin_ptr, rotary_dim)

    # 3. Write K, V to linear cache
    kv_cache.write_kv(layer_idx, pos, k_ptr, v_ptr)

    # 4. Standard causal attention over all past positions
    var valid_len = pos + 1
    var kv_ptrs = kv_cache.get_kv_ptrs(layer_idx)
    var layer_k_ptr = kv_ptrs.first
    var layer_v_ptr = kv_ptrs.second

    var heads_per_kv = num_heads // num_kv_heads
    var scale = 1.0 / sqrt(Float32(head_dim))
    var attn_out_ptr = scratch_ptr + q_size + kv_size + kv_size

    for h in range(num_heads):
        var kv_h = h // heads_per_kv
        var q_head_ptr = q_ptr + h * head_dim
        var scores_ptr = attn_out_ptr + q_size + h * max_seq_len

        for t in range(valid_len):
            var k_head_ptr = layer_k_ptr + t * kv_size + kv_h * head_dim
            var score: Float32 = 0.0
            for d in range(head_dim):
                score += q_head_ptr.load(d) * k_head_ptr.load(d)
            scores_ptr.store(t, score * scale)

        softmax(scores_ptr, valid_len)

        # Weighted sum of V
        var out_head_ptr = attn_out_ptr + h * head_dim
        for d in range(head_dim):
            out_head_ptr.store(d, 0.0)
        for t in range(valid_len):
            var v_head_ptr = layer_v_ptr + t * kv_size + kv_h * head_dim
            var prob = scores_ptr.load(t)
            for d in range(head_dim):
                out_head_ptr.store(d, out_head_ptr.load(d) + prob * v_head_ptr.load(d))

    # 5. Output projection
    _gemm_dispatch(out_ptr, attn_out_ptr, weights.o_proj, 1, q_size, hidden_size)


@always_inline
fn forward_vision_attention(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],  # [num_tokens, hidden_size]
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],    # [num_tokens, hidden_size]
    weights: VisionLayerWeights,
    num_tokens: Int,
    hidden_size: Int,
    num_heads: Int,
    head_dim: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin],
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
    mat_mat_mul(q_ptr, x_ptr, weights.q_proj.ptr, num_tokens, hidden_size, num_heads * head_dim)
    mat_mat_mul(k_ptr, x_ptr, weights.k_proj.ptr, num_tokens, hidden_size, num_heads * head_dim)
    mat_mat_mul(v_ptr, x_ptr, weights.v_proj.ptr, num_tokens, hidden_size, num_heads * head_dim)

    var scale = 1.0 / sqrt(Float32(head_dim))
    var attn_out_ptr = scratch_ptr + q_size + kv_size + kv_size
    var scores_ptr = attn_out_ptr + q_size  # [num_heads, num_tokens, num_tokens]

    # Per-head attention
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
            softmax(s_row, num_tokens)

            # Weighted sum of values
            var out_head = attn_out_ptr + qi * num_heads * head_dim + h * head_dim
            for d in range(head_dim):
                out_head.store(d, 0.0)
            for vi in range(num_tokens):
                var v_head = v_ptr + vi * num_heads * head_dim + h * head_dim
                var prob = s_row.load(vi)
                for d in range(head_dim):
                    out_head.store(d, out_head.load(d) + prob * v_head.load(d))

    # Output projection: [num_tokens, num_heads*head_dim] @ [hidden_size, num_heads*head_dim]^T
    mat_mat_mul(out_ptr, attn_out_ptr, weights.o_proj.ptr, num_tokens, num_heads * head_dim, hidden_size)


@always_inline
fn forward_vision_layer(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],  # [num_tokens, hidden_size]
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],    # [num_tokens, hidden_size]
    weights: VisionLayerWeights,
    num_tokens: Int,
    hidden_size: Int,
    num_heads: Int,
    head_dim: Int,
    intermediate_size: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin],
):
    """Single vision transformer layer: LayerNorm + bidirectional attn + residual + LayerNorm + GELU MLP + residual."""
    var total = num_tokens * hidden_size

    # Pre-attention LayerNorm (reuse rms_norm — SigLIP weights are scale-only)
    var norm1_ptr = scratch_ptr
    for t in range(num_tokens):
        rms_norm(norm1_ptr + t * hidden_size, x_ptr + t * hidden_size, weights.layer_norm1.ptr, hidden_size, 1e-6)

    # Bidirectional attention
    var attn_out_ptr = scratch_ptr + total
    var attn_scratch = scratch_ptr + total * 2
    forward_vision_attention(
        attn_out_ptr, norm1_ptr, weights,
        num_tokens, hidden_size, num_heads, head_dim, attn_scratch,
    )

    # Attention residual
    var residual_ptr = scratch_ptr + total * 2
    for i in range(total):
        residual_ptr.store(i, x_ptr.load(i) + attn_out_ptr.load(i))

    # Pre-MLP LayerNorm
    var norm2_ptr = scratch_ptr + total * 3
    for t in range(num_tokens):
        rms_norm(norm2_ptr + t * hidden_size, residual_ptr + t * hidden_size, weights.layer_norm2.ptr, hidden_size, 1e-6)

    # Vision MLP: fc1 → GELU → fc2 (NOT GEGLU)
    var fc1_out_ptr = scratch_ptr + total * 4
    mat_mat_mul(fc1_out_ptr, norm2_ptr, weights.fc1.ptr, num_tokens, hidden_size, intermediate_size)

    var gelu_out_ptr = scratch_ptr + total * 4 + num_tokens * intermediate_size
    for t in range(num_tokens):
        gelu(gelu_out_ptr + t * intermediate_size, fc1_out_ptr + t * intermediate_size, intermediate_size)

    var mlp_out_ptr = scratch_ptr + total * 5 + num_tokens * intermediate_size
    mat_mat_mul(mlp_out_ptr, gelu_out_ptr, weights.fc2.ptr, num_tokens, intermediate_size, hidden_size)

    # MLP residual
    for i in range(total):
        out_ptr.store(i, residual_ptr.load(i) + mlp_out_ptr.load(i))


@always_inline
fn forward_vision_encoder(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    patches_ptr: UnsafePointer[Float32, MutExternalOrigin],  # [num_patches, patch_dim]
    weights: VisionModelWeights,
    num_patches: Int,
    grid_h: Int,
    grid_w: Int,
    vision_hidden_size: Int,
    vision_num_heads: Int,
    vision_head_dim: Int,
    vision_intermediate_size: Int,
    decoder_hidden_size: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin],
):
    """Full SigLIP vision encoder: patch embed → position embed → N layers → post-norm → avg pool → project."""
    var patch_dim = weights.patch_embedding.shape_1
    var total = num_patches * vision_hidden_size

    # 1. Patch embedding: [num_patches, patch_dim] @ [vision_hidden, patch_dim]^T → [num_patches, vision_hidden]
    var embedded_ptr = scratch_ptr
    mat_mat_mul(embedded_ptr, patches_ptr, weights.patch_embedding.ptr, num_patches, patch_dim, vision_hidden_size)

    # 2. Add position embeddings (learned, [max_patches, vision_hidden])
    for i in range(total):
        embedded_ptr.store(i, embedded_ptr.load(i) + weights.position_embedding.ptr.load(i))

    # 3. Vision transformer layers
    var current_ptr = embedded_ptr
    var next_ptr = scratch_ptr + total
    var layer_scratch = scratch_ptr + total * 2

    var num_layers = len(weights.layers)
    for l in range(num_layers):
        forward_vision_layer(
            next_ptr, current_ptr, weights.layers[l],
            num_patches, vision_hidden_size, vision_num_heads, vision_head_dim,
            vision_intermediate_size, layer_scratch,
        )
        # Swap
        for i in range(total):
            current_ptr.store(i, next_ptr.load(i))

    # 4. Post-LayerNorm
    var norm_ptr = next_ptr
    for t in range(num_patches):
        rms_norm(norm_ptr + t * vision_hidden_size, current_ptr + t * vision_hidden_size,
                 weights.post_norm.ptr, vision_hidden_size, 1e-6)

    # 5. Average pooling (3×3)
    var pool_kernel = 3
    var pool_h = grid_h // pool_kernel
    var pool_w = grid_w // pool_kernel
    var pooled_tokens = pool_h * pool_w
    var pooled_ptr = scratch_ptr + total * 2
    average_pool_2d(pooled_ptr, norm_ptr, grid_h, grid_w, vision_hidden_size, pool_kernel)

    # 6. Vision projection → decoder hidden dim
    mat_mat_mul(out_ptr, pooled_ptr, weights.projection.ptr, pooled_tokens, vision_hidden_size, decoder_hidden_size)


@always_inline
fn forward_audio_encoder(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    features_ptr: UnsafePointer[Float32, MutExternalOrigin],  # [n_mels, num_frames] (flattened)
    weights: AudioTowerWeights,
    num_frames: Int,
    n_mels: Int,
    audio_hidden_size: Int,
    audio_num_heads: Int,
    audio_head_dim: Int,
    audio_intermediate_size: Int,
    decoder_hidden_size: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin],
):
    """Audio encoder: conv feature extraction → position embed → transformer layers → post-norm → projection.

    Reuses forward_vision_layer for the transformer layers (same bidirectional attention + GELU MLP).
    Conv layers downsample mel frames into audio_hidden_size-dim tokens.
    """
    var num_conv = len(weights.conv_weights)
    var num_tokens = num_frames

    # 1. Conv feature extraction (simplified: treat as linear projections over frames)
    # First conv: [n_mels] → [audio_hidden_size] per frame
    var conv_out_ptr = scratch_ptr
    if num_conv > 0:
        mat_mat_mul(conv_out_ptr, features_ptr, weights.conv_weights[0].ptr, num_tokens, n_mels, audio_hidden_size)
        # Subsequent conv layers: [audio_hidden_size] → [audio_hidden_size] with stride-2 downsampling
        for c in range(1, num_conv):
            var new_tokens = num_tokens // 2
            if new_tokens == 0:
                new_tokens = 1
            var next_conv_ptr = conv_out_ptr + num_tokens * audio_hidden_size
            # Stride-2: take every other token, project
            for t in range(new_tokens):
                var src_idx = t * 2
                vec_mat_mul(
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
        mat_mat_mul(conv_out_ptr, features_ptr, weights.projection.ptr, num_tokens, n_mels, audio_hidden_size)

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
        forward_vision_layer(
            next_ptr, current_ptr, weights.layers[l],
            num_tokens, audio_hidden_size, audio_num_heads, audio_head_dim,
            audio_intermediate_size, layer_scratch,
        )
        for i in range(total):
            current_ptr.store(i, next_ptr.load(i))

    # 4. Post-LayerNorm
    var norm_ptr = next_ptr
    for t in range(num_tokens):
        rms_norm(norm_ptr + t * audio_hidden_size, current_ptr + t * audio_hidden_size,
                 weights.post_norm.ptr, audio_hidden_size, 1e-6)

    # 5. Projection → decoder hidden dim
    mat_mat_mul(out_ptr, norm_ptr, weights.projection.ptr, num_tokens, audio_hidden_size, decoder_hidden_size)


@always_inline
fn forward_mlp(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],  # [batch_size, hidden_size]
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],  # [batch_size, hidden_size]
    weights: LayerWeights,
    hidden_size: Int,
    intermediate_size: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin],  # temp memory
    batch_size: Int = 1,
):
    """Computes the feed-forward network (MLP) block for a standard transformer layer.

    Projects the input hidden state to an intermediate size through gate and up projections, applies the GEGLU activation function, and then down-projects back to the hidden size.
    """
    var gate_ptr = scratch_ptr
    var up_ptr = scratch_ptr + batch_size * intermediate_size
    var geglu_out_ptr = scratch_ptr + batch_size * intermediate_size * 2

    _gemm_dispatch(gate_ptr, x_ptr, weights.gate_proj, batch_size, hidden_size, intermediate_size)
    _gemm_dispatch(up_ptr, x_ptr, weights.up_proj, batch_size, hidden_size, intermediate_size)

    for b in range(batch_size):
        geglu(
            geglu_out_ptr + b * intermediate_size,
            gate_ptr + b * intermediate_size,
            up_ptr + b * intermediate_size,
            intermediate_size,
        )

    _gemm_dispatch(out_ptr, geglu_out_ptr, weights.down_proj, batch_size, intermediate_size, hidden_size)


@always_inline
fn forward_gemma4_layer(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],  # [hidden_size]
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],    # [hidden_size]
    weights: LayerWeights,
    layer_idx: Int,
    pos: Int,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    intermediate_size: Int,
    kv_cache: KVCache,
    rope_tables: RoPETables,
    k_eq_v: Bool,
    max_seq_len: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin],
):
    """Executes a single Gemma 4 transformer layer with attention type dispatch.

    Applies pre-attention RMSNorm, dispatches to sliding or full attention based on layer type,
    adds the attention residual, applies pre-MLP RMSNorm, runs the MLP, and adds the MLP residual.
    """
    # Pre-attention norm
    var norm_x_ptr = scratch_ptr
    rms_norm(norm_x_ptr, x_ptr, weights.input_layernorm.ptr, hidden_size, 1e-6)

    # Attention dispatch
    var attn_out_ptr = scratch_ptr + hidden_size
    var attn_scratch_ptr = scratch_ptr + hidden_size * 2
    if kv_cache.layer_types[layer_idx] == LAYER_TYPE_SLIDING:
        forward_sliding_attention(
            attn_out_ptr, norm_x_ptr, weights, layer_idx, pos,
            hidden_size, num_heads, num_kv_heads, head_dim,
            kv_cache, rope_tables, k_eq_v, attn_scratch_ptr,
        )
    else:
        forward_full_attention(
            attn_out_ptr, norm_x_ptr, weights, layer_idx, pos,
            hidden_size, num_heads, num_kv_heads, head_dim,
            kv_cache, rope_tables, k_eq_v, max_seq_len, attn_scratch_ptr,
        )

    # Post-attention norm
    var post_attn_ptr = scratch_ptr + hidden_size * 2
    rms_norm(post_attn_ptr, attn_out_ptr, weights.post_attention_layernorm.ptr, hidden_size, 1e-6)

    # Attention residual
    var residual_ptr = scratch_ptr + hidden_size * 3
    for i in range(hidden_size):
        residual_ptr.store(i, x_ptr.load(i) + post_attn_ptr.load(i))

    # Pre-MLP norm
    var norm_residual_ptr = scratch_ptr + hidden_size * 4
    rms_norm(norm_residual_ptr, residual_ptr, weights.pre_feedforward_layernorm.ptr, hidden_size, 1e-6)

    # MLP
    var mlp_out_ptr = scratch_ptr + hidden_size * 5
    var mlp_scratch_ptr = scratch_ptr + hidden_size * 6
    forward_mlp(mlp_out_ptr, norm_residual_ptr, weights, hidden_size, intermediate_size, mlp_scratch_ptr)

    # Post-MLP norm
    var post_mlp_ptr = scratch_ptr + hidden_size * 7
    rms_norm(post_mlp_ptr, mlp_out_ptr, weights.post_feedforward_layernorm.ptr, hidden_size, 1e-6)

    # MLP residual
    for i in range(hidden_size):
        out_ptr.store(i, residual_ptr.load(i) + post_mlp_ptr.load(i))


@always_inline
fn forward_gemma4_step(
    out_logits_ptr: UnsafePointer[Float32, MutExternalOrigin],  # [vocab_size]
    token_id: Int,
    pos: Int,
    model: ModelWeights,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    intermediate_size: Int,
    vocab_size: Int,
    kv_cache: KVCache,
    rope_tables: RoPETables,
    k_eq_v: Bool,
    max_seq_len: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin],
):
    """Executes a single autoregressive generation step for Gemma 4.

    Embeds the token, passes through all transformer layers with hybrid attention dispatch,
    applies final RMSNorm, and projects to logits.
    """
    var num_layers = len(model.layers)
    var current_state = scratch_ptr
    var next_state = scratch_ptr + hidden_size
    var layer_scratch = scratch_ptr + hidden_size * 2

    # Embed and scale
    var emb_scale = sqrt(Float32(hidden_size))
    model.get_embedding(token_id, current_state)
    for i in range(hidden_size):
        current_state.store(i, current_state.load(i) * emb_scale)

    # Layer loop with attention type dispatch
    for l in range(num_layers):
        forward_gemma4_layer(
            next_state, current_state, model.layers[l], l, pos,
            hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size,
            kv_cache, rope_tables, k_eq_v, max_seq_len, layer_scratch,
        )
        # Swap states
        for i in range(hidden_size):
            current_state.store(i, next_state.load(i))

    # Final norm + LM head
    var norm_out = next_state
    rms_norm(norm_out, current_state, model.norm.ptr, hidden_size, 1e-6)
    vec_mat_mul(out_logits_ptr, norm_out, model.lm_head.ptr, hidden_size, vocab_size)


@always_inline
fn forward_gemma4_step_with_embedding(
    out_logits_ptr: UnsafePointer[Float32, MutExternalOrigin],  # [vocab_size]
    embedding_ptr: UnsafePointer[Float32, MutExternalOrigin],   # [hidden_size] — pre-computed embedding
    pos: Int,
    model: ModelWeights,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    intermediate_size: Int,
    vocab_size: Int,
    kv_cache: KVCache,
    rope_tables: RoPETables,
    k_eq_v: Bool,
    max_seq_len: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin],
):
    """Like forward_gemma4_step but uses a pre-computed embedding instead of token lookup.

    Used for vision token injection during prefill.
    """
    var num_layers = len(model.layers)
    var current_state = scratch_ptr
    var next_state = scratch_ptr + hidden_size
    var layer_scratch = scratch_ptr + hidden_size * 2

    # Use provided embedding directly (already scaled by caller)
    for i in range(hidden_size):
        current_state.store(i, embedding_ptr.load(i))

    # Layer loop (same as forward_gemma4_step)
    for l in range(num_layers):
        forward_gemma4_layer(
            next_state, current_state, model.layers[l], l, pos,
            hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size,
            kv_cache, rope_tables, k_eq_v, max_seq_len, layer_scratch,
        )
        for i in range(hidden_size):
            current_state.store(i, next_state.load(i))

    # Final norm + LM head (step_with_embedding path)
    var norm_out = next_state
    rms_norm(norm_out, current_state, model.norm.ptr, hidden_size, 1e-6)
    vec_mat_mul(out_logits_ptr, norm_out, model.lm_head.ptr, hidden_size, vocab_size)


# ── PLE (Per-Layer Embedding) for E2B/E4B ────────────────────────────────

@always_inline
fn forward_ple_input(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    token_id: Int,
    ple_weights: PLELayerWeights,
    hidden_size: Int,
    ple_dim: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin],
):
    """Inject per-layer embedding into hidden state: embed → project → norm → add."""
    var embed_ptr = scratch_ptr
    var proj_ptr = scratch_ptr + ple_dim
    var emb_src = ple_weights.per_layer_embedding.ptr + token_id * ple_dim
    for i in range(ple_dim):
        embed_ptr.store(i, emb_src.load(i))
    vec_mat_mul(proj_ptr, embed_ptr, ple_weights.per_layer_projection.ptr, ple_dim, hidden_size)
    var normed_ptr = scratch_ptr + ple_dim + hidden_size
    rms_norm(normed_ptr, proj_ptr, ple_weights.per_layer_norm.ptr, hidden_size, 1e-6)
    for i in range(hidden_size):
        out_ptr.store(i, out_ptr.load(i) + normed_ptr.load(i))


@always_inline
fn forward_gemma4_ple_step(
    out_logits_ptr: UnsafePointer[Float32, MutExternalOrigin],
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
    kv_cache: KVCache,
    rope_tables: RoPETables,
    k_eq_v: Bool,
    max_seq_len: Int,
    kv_sharing_map_ptr: UnsafePointer[Int64, MutExternalOrigin],
    num_kv_sharing_layers: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin],
):
    """E2B/E4B forward step with PLE injection and optional shared-KV attention."""
    var num_layers = len(model.layers)
    var current_state = scratch_ptr
    var next_state = scratch_ptr + hidden_size
    var layer_scratch = scratch_ptr + hidden_size * 2
    var emb_scale = sqrt(Float32(hidden_size))
    model.get_embedding(token_id, current_state)
    for i in range(hidden_size):
        current_state.store(i, current_state.load(i) * emb_scale)
    for l in range(num_layers):
        if model.has_ple and l < len(model.ple_layers):
            forward_ple_input(current_state, token_id, model.ple_layers[l], hidden_size, ple_dim, layer_scratch)
        var source_layer = -1
        if num_kv_sharing_layers > 0 and l < num_kv_sharing_layers:
            source_layer = Int(kv_sharing_map_ptr.load(l))
        if source_layer >= 0:
            forward_gemma4_layer(next_state, current_state, model.layers[l], source_layer, pos, hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, kv_cache, rope_tables, k_eq_v, max_seq_len, layer_scratch)
        else:
            forward_gemma4_layer(next_state, current_state, model.layers[l], l, pos, hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, kv_cache, rope_tables, k_eq_v, max_seq_len, layer_scratch)
        for i in range(hidden_size):
            current_state.store(i, next_state.load(i))
    var norm_out_ple = next_state
    rms_norm(norm_out_ple, current_state, model.norm.ptr, hidden_size, 1e-6)
    vec_mat_mul(out_logits_ptr, norm_out_ple, model.lm_head.ptr, hidden_size, vocab_size)


# ── MoE (Mixture of Experts) for 26B ─────────────────────────────────────

@always_inline
fn forward_moe_router(
    expert_indices_ptr: UnsafePointer[Int32, MutExternalOrigin],
    expert_weights_ptr: UnsafePointer[Float32, MutExternalOrigin],
    hidden_ptr: UnsafePointer[Float32, MutExternalOrigin],
    router_weight: TensorInfo,
    hidden_size: Int,
    num_experts: Int,
    k: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin],
):
    """Route hidden state to top-k experts: logits → softmax → top_k → renormalize."""
    var logits_ptr = scratch_ptr
    vec_mat_mul(logits_ptr, hidden_ptr, router_weight.ptr, hidden_size, num_experts)
    softmax(logits_ptr, num_experts)
    top_k(logits_ptr, k, num_experts, expert_indices_ptr, expert_weights_ptr)
    var weight_sum: Float32 = 0.0
    for i in range(k):
        weight_sum += expert_weights_ptr.load(i)
    if weight_sum > 0.0:
        var inv_sum = 1.0 / weight_sum
        for i in range(k):
            expert_weights_ptr.store(i, expert_weights_ptr.load(i) * inv_sum)


@always_inline
fn forward_moe_experts(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    hidden_ptr: UnsafePointer[Float32, MutExternalOrigin],
    expert_indices_ptr: UnsafePointer[Int32, MutExternalOrigin],
    expert_weights_ptr: UnsafePointer[Float32, MutExternalOrigin],
    experts: List[MoEExpertWeights],
    k: Int,
    hidden_size: Int,
    intermediate_size: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin],
):
    """Execute selected experts and compute weighted sum."""
    var gate_ptr = scratch_ptr
    var up_ptr = scratch_ptr + intermediate_size
    var geglu_out_ptr = scratch_ptr + intermediate_size * 2
    var expert_out_ptr = scratch_ptr + intermediate_size * 3
    for i in range(hidden_size):
        out_ptr.store(i, 0.0)
    for sel in range(k):
        var idx = Int(expert_indices_ptr.load(sel))
        var weight = expert_weights_ptr.load(sel)
        vec_mat_mul(gate_ptr, hidden_ptr, experts[idx].gate_proj.ptr, hidden_size, intermediate_size)
        vec_mat_mul(up_ptr, hidden_ptr, experts[idx].up_proj.ptr, hidden_size, intermediate_size)
        geglu(geglu_out_ptr, gate_ptr, up_ptr, intermediate_size)
        vec_mat_mul(expert_out_ptr, geglu_out_ptr, experts[idx].down_proj.ptr, intermediate_size, hidden_size)
        for i in range(hidden_size):
            out_ptr.store(i, out_ptr.load(i) + weight * expert_out_ptr.load(i))


@always_inline
fn forward_moe_layer(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],
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
    kv_cache: KVCache,
    rope_tables: RoPETables,
    max_seq_len: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin],
):
    """Single MoE transformer layer: attention (K=V) + MoE block."""
    var norm_x_ptr = scratch_ptr
    rms_norm(norm_x_ptr, x_ptr, weights.input_layernorm.ptr, hidden_size, 1e-6)
    var q_size = num_heads * head_dim
    var kv_size = num_kv_heads * head_dim
    var attn_out_ptr = scratch_ptr + hidden_size
    var attn_scratch_ptr = scratch_ptr + hidden_size * 2
    var q_ptr = attn_scratch_ptr
    var k_ptr = attn_scratch_ptr + q_size
    var v_ptr = attn_scratch_ptr + q_size + kv_size
    _gemm_dispatch(q_ptr, norm_x_ptr, weights.q_proj, 1, hidden_size, q_size)
    _gemm_dispatch(k_ptr, norm_x_ptr, weights.k_proj, 1, hidden_size, kv_size)
    _gemm_dispatch(v_ptr, norm_x_ptr, weights.k_proj, 1, hidden_size, kv_size)  # K=V
    if weights.q_norm.ptr != UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=0):
        for h in range(num_heads):
            rms_norm(q_ptr + h * head_dim, q_ptr + h * head_dim, weights.q_norm.ptr, head_dim, 1e-6)
    if weights.k_norm.ptr != UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=0):
        for h in range(num_kv_heads):
            rms_norm(k_ptr + h * head_dim, k_ptr + h * head_dim, weights.k_norm.ptr, head_dim, 1e-6)
    if kv_cache.layer_types[layer_idx] == LAYER_TYPE_SLIDING:
        var sliding_freqs = rope_tables.get_sliding_freqs(pos)
        var cos_ptr = sliding_freqs.first
        var sin_ptr = sliding_freqs.second
        for h in range(num_heads):
            rope_rotate(q_ptr + h * head_dim, cos_ptr, sin_ptr, head_dim)
        for h in range(num_kv_heads):
            rope_rotate(k_ptr + h * head_dim, cos_ptr, sin_ptr, head_dim)
    else:
        var rotary_dim = rope_tables.rotary_dim
        var full_freqs = rope_tables.get_full_freqs(pos)
        var cos_ptr = full_freqs.first
        var sin_ptr = full_freqs.second
        for h in range(num_heads):
            rope_rotate(q_ptr + h * head_dim, cos_ptr, sin_ptr, rotary_dim)
        for h in range(num_kv_heads):
            rope_rotate(k_ptr + h * head_dim, cos_ptr, sin_ptr, rotary_dim)
    kv_cache.write_kv(layer_idx, pos, k_ptr, v_ptr)
    var attn_range = kv_cache.get_attention_range(layer_idx, pos)
    var valid_len = attn_range.first
    var cache_size = attn_range.second
    var kv_ptrs = kv_cache.get_kv_ptrs(layer_idx)
    var layer_k_ptr = kv_ptrs.first
    var layer_v_ptr = kv_ptrs.second
    var heads_per_kv = num_heads // num_kv_heads
    var scale = 1.0 / sqrt(Float32(head_dim))
    var attn_weighted_ptr = attn_scratch_ptr + q_size + kv_size + kv_size
    for h in range(num_heads):
        var kv_h = h // heads_per_kv
        var q_head_ptr = q_ptr + h * head_dim
        var scores_ptr = attn_weighted_ptr + q_size + h * max_seq_len
        for t in range(valid_len):
            var k_head_ptr = layer_k_ptr + t * kv_size + kv_h * head_dim
            var score: Float32 = 0.0
            for d in range(head_dim):
                score += q_head_ptr.load(d) * k_head_ptr.load(d)
            scores_ptr.store(t, score * scale)
        softmax(scores_ptr, valid_len)
        var out_head_ptr = attn_weighted_ptr + h * head_dim
        for d in range(head_dim):
            out_head_ptr.store(d, 0.0)
        for t in range(valid_len):
            var v_head_ptr = layer_v_ptr + t * kv_size + kv_h * head_dim
            var prob = scores_ptr.load(t)
            for d in range(head_dim):
                out_head_ptr.store(d, out_head_ptr.load(d) + prob * v_head_ptr.load(d))
    _gemm_dispatch(attn_out_ptr, attn_weighted_ptr, weights.o_proj, 1, q_size, hidden_size)
    var post_attn_ptr = scratch_ptr + hidden_size * 2
    rms_norm(post_attn_ptr, attn_out_ptr, weights.post_attention_layernorm.ptr, hidden_size, 1e-6)
    var residual_ptr = scratch_ptr + hidden_size * 3
    for i in range(hidden_size):
        residual_ptr.store(i, x_ptr.load(i) + post_attn_ptr.load(i))
    var norm_residual_ptr = scratch_ptr + hidden_size * 4
    rms_norm(norm_residual_ptr, residual_ptr, weights.pre_feedforward_layernorm.ptr, hidden_size, 1e-6)
    var moe_out_ptr = scratch_ptr + hidden_size * 5
    var moe_scratch = scratch_ptr + hidden_size * 6
    var expert_indices_ptr = UnsafePointer[Int32, MutExternalOrigin](unsafe_from_address=Int(moe_scratch))
    var expert_weights_ptr = moe_scratch + moe_top_k
    var router_scratch = moe_scratch + moe_top_k * 2
    forward_moe_router(expert_indices_ptr, expert_weights_ptr, norm_residual_ptr, weights.router, hidden_size, num_experts, moe_top_k, router_scratch)
    var expert_scratch = router_scratch + num_experts
    forward_moe_experts(moe_out_ptr, norm_residual_ptr, expert_indices_ptr, expert_weights_ptr, weights.experts, moe_top_k, hidden_size, moe_intermediate_size, expert_scratch)
    var post_moe_ptr = scratch_ptr + hidden_size * 7
    rms_norm(post_moe_ptr, moe_out_ptr, weights.post_feedforward_layernorm.ptr, hidden_size, 1e-6)
    for i in range(hidden_size):
        out_ptr.store(i, residual_ptr.load(i) + post_moe_ptr.load(i))


@always_inline
fn forward_gemma4_moe_step(
    out_logits_ptr: UnsafePointer[Float32, MutExternalOrigin],
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
    kv_cache: KVCache,
    rope_tables: RoPETables,
    max_seq_len: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin],
):
    """Full 26B MoE forward step: embed → MoE layers → norm → logits."""
    var num_layers = len(model.layers)
    var current_state = scratch_ptr
    var next_state = scratch_ptr + hidden_size
    var layer_scratch = scratch_ptr + hidden_size * 2
    var emb_scale = sqrt(Float32(hidden_size))
    model.get_embedding(token_id, current_state)
    for i in range(hidden_size):
        current_state.store(i, current_state.load(i) * emb_scale)
    for l in range(num_layers):
        forward_moe_layer(next_state, current_state, model.layers[l], l, pos, hidden_size, num_heads, num_kv_heads, head_dim, num_experts, moe_top_k, moe_intermediate_size, kv_cache, rope_tables, max_seq_len, layer_scratch)
        for i in range(hidden_size):
            current_state.store(i, next_state.load(i))
    var norm_out_moe = next_state
    rms_norm(norm_out_moe, current_state, model.norm.ptr, hidden_size, 1e-6)
    vec_mat_mul(out_logits_ptr, norm_out_moe, model.lm_head.ptr, hidden_size, vocab_size)
