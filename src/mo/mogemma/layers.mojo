from std.memory import UnsafePointer
from std.math import sqrt, erf, tanh
from mogemma.model import (
    LayerWeights,
    ModelWeights,
    TensorInfo,
    KVCache,
    RoPETables,
    LAYER_TYPE_SLIDING,
    LAYER_TYPE_FULL,
)
from mogemma.ops import vec_mat_mul, rope_rotate, softmax, rms_norm, geglu, mat_mat_mul, mat_mat_mul_i8


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
    var cos_ptr: UnsafePointer[Float32, MutExternalOrigin]
    var sin_ptr: UnsafePointer[Float32, MutExternalOrigin]
    (cos_ptr, sin_ptr) = rope_tables.get_sliding_freqs(pos)
    for h in range(num_heads):
        rope_rotate(q_ptr + h * head_dim, cos_ptr, sin_ptr, head_dim)
    for h in range(num_kv_heads):
        rope_rotate(k_ptr + h * head_dim, cos_ptr, sin_ptr, head_dim)

    # 3. Write K, V to ring buffer
    kv_cache.write_kv(layer_idx, pos, k_ptr, v_ptr)

    # 4. Compute attention within sliding window
    var valid_len: Int
    var cache_size: Int
    (valid_len, cache_size) = kv_cache.get_attention_range(layer_idx, pos)
    var layer_k_ptr: UnsafePointer[Float32, MutExternalOrigin]
    var layer_v_ptr: UnsafePointer[Float32, MutExternalOrigin]
    (layer_k_ptr, layer_v_ptr) = kv_cache.get_kv_ptrs(layer_idx)

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
    var cos_ptr: UnsafePointer[Float32, MutExternalOrigin]
    var sin_ptr: UnsafePointer[Float32, MutExternalOrigin]
    (cos_ptr, sin_ptr) = rope_tables.get_full_freqs(pos)
    for h in range(num_heads):
        # Only rotate first rotary_dim dimensions of each head
        rope_rotate(q_ptr + h * head_dim, cos_ptr, sin_ptr, rotary_dim)
    for h in range(num_kv_heads):
        rope_rotate(k_ptr + h * head_dim, cos_ptr, sin_ptr, rotary_dim)

    # 3. Write K, V to linear cache
    kv_cache.write_kv(layer_idx, pos, k_ptr, v_ptr)

    # 4. Standard causal attention over all past positions
    var valid_len = pos + 1
    var layer_k_ptr: UnsafePointer[Float32, MutExternalOrigin]
    var layer_v_ptr: UnsafePointer[Float32, MutExternalOrigin]
    (layer_k_ptr, layer_v_ptr) = kv_cache.get_kv_ptrs(layer_idx)

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
