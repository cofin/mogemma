from memory import UnsafePointer
from math import sqrt, erf, tanh
from mogemma.model import (
    LayerWeights,
    ModelWeights,
    AltUpWeights,
    LaurelWeights,
    PerLayerMapWeights,
    NanoLayerWeights,
    NanoModelWeights,
    TensorInfo,
)
from mogemma.ops import vec_mat_mul, rope_rotate, softmax, rms_norm, geglu

@always_inline
fn forward_attention(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],
    weights: LayerWeights,
    pos: Int,  # current token index in sequence
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    freqs_cos_ptr: UnsafePointer[Float32, MutExternalOrigin], # for this pos
    freqs_sin_ptr: UnsafePointer[Float32, MutExternalOrigin], # for this pos
    kv_cache_k_ptr: UnsafePointer[Float32, MutExternalOrigin], # [max_seq_len, num_kv_heads, head_dim]
    kv_cache_v_ptr: UnsafePointer[Float32, MutExternalOrigin], # [max_seq_len, num_kv_heads, head_dim]
    max_seq_len: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin] # temporary memory
):
    # 1. Project Q, K, V
    var q_size = num_heads * head_dim
    var kv_size = num_kv_heads * head_dim
    var q_ptr = scratch_ptr
    var k_ptr = scratch_ptr + q_size
    var v_ptr = scratch_ptr + q_size + kv_size
    
    vec_mat_mul(q_ptr, x_ptr, weights.q_proj.ptr, hidden_size, q_size)
    vec_mat_mul(k_ptr, x_ptr, weights.k_proj.ptr, hidden_size, kv_size)
    vec_mat_mul(v_ptr, x_ptr, weights.v_proj.ptr, hidden_size, kv_size)

    # 1b. Apply per-head QK norms (if weights are present)
    if weights.q_norm.ptr != UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=0):
        for h in range(num_heads):
            rms_norm(q_ptr + h * head_dim, q_ptr + h * head_dim, weights.q_norm.ptr, head_dim, 1e-6)
    if weights.k_norm.ptr != UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=0):
        for h in range(num_kv_heads):
            rms_norm(k_ptr + h * head_dim, k_ptr + h * head_dim, weights.k_norm.ptr, head_dim, 1e-6)

    # 2. Apply RoPE to Q and K
    for h in range(num_heads):
        rope_rotate(q_ptr + h * head_dim, freqs_cos_ptr, freqs_sin_ptr, head_dim)
    for h in range(num_kv_heads):
        rope_rotate(k_ptr + h * head_dim, freqs_cos_ptr, freqs_sin_ptr, head_dim)
        
    # 3. Write K, V to KV Cache
    var kv_offset = pos * kv_size
    for i in range(kv_size):
        kv_cache_k_ptr.store(kv_offset + i, k_ptr.load(i))
        kv_cache_v_ptr.store(kv_offset + i, v_ptr.load(i))
        
    # 4. Attention for each head
    var heads_per_kv = num_heads // num_kv_heads
    var scale = 1.0 / sqrt(Float32(head_dim))
    var attn_out_ptr = scratch_ptr + q_size + kv_size + kv_size # use memory after V
    
    for h in range(num_heads):
        var kv_h = h // heads_per_kv
        var q_head_ptr = q_ptr + h * head_dim
        var scores_ptr = attn_out_ptr + num_heads * head_dim # temporary scores for this head
        
        # Calculate scores for past tokens up to `pos`
        for t in range(pos + 1):
            var k_head_ptr = kv_cache_k_ptr + t * kv_size + kv_h * head_dim
            var score: Float32 = 0.0
            for d in range(head_dim):
                score += q_head_ptr.load(d) * k_head_ptr.load(d)
            scores_ptr.store(t, score * scale)
            
        # Softmax
        softmax(scores_ptr, pos + 1)
        
        # Weighted sum of V
        var out_head_ptr = attn_out_ptr + h * head_dim
        for d in range(head_dim):
            out_head_ptr.store(d, 0.0)
            
        for t in range(pos + 1):
            var v_head_ptr = kv_cache_v_ptr + t * kv_size + kv_h * head_dim
            var prob = scores_ptr.load(t)
            for d in range(head_dim):
                var acc = out_head_ptr.load(d)
                out_head_ptr.store(d, acc + prob * v_head_ptr.load(d))
                
    # 5. Output Projection
    vec_mat_mul(out_ptr, attn_out_ptr, weights.o_proj.ptr, q_size, hidden_size)


@always_inline
fn forward_mlp(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],
    weights: LayerWeights,
    hidden_size: Int,
    intermediate_size: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin] # temp memory
):
    var gate_ptr = scratch_ptr
    var up_ptr = scratch_ptr + intermediate_size
    var geglu_out_ptr = scratch_ptr + intermediate_size * 2
    
    vec_mat_mul(gate_ptr, x_ptr, weights.gate_proj.ptr, hidden_size, intermediate_size)
    vec_mat_mul(up_ptr, x_ptr, weights.up_proj.ptr, hidden_size, intermediate_size)
    
    geglu(geglu_out_ptr, gate_ptr, up_ptr, intermediate_size)
    
    vec_mat_mul(out_ptr, geglu_out_ptr, weights.down_proj.ptr, intermediate_size, hidden_size)

@always_inline
fn _apply_nano_activation_sparsity(
    gate_ptr: UnsafePointer[Float32, MutExternalOrigin],
    size: Int,
    sparsity: Float32,
):
    if sparsity <= 0.0:
        return

    # For Gemma3n default activation_sparsity=0.95, z = N(0,1)^{-1}(0.95).
    var z_score: Float32 = 1.6448536269514722
    var sum: Float32 = 0.0
    for i in range(size):
        sum += gate_ptr.load(i)
    var mean = sum / Float32(size)

    var var_sum: Float32 = 0.0
    for i in range(size):
        var d = gate_ptr.load(i) - mean
        var_sum += d * d
    var std = sqrt(var_sum / Float32(size))
    var cutoff = mean + std * z_score

    for i in range(size):
        var v = gate_ptr.load(i) - cutoff
        if v < 0.0:
            v = 0.0
        gate_ptr.store(i, v)

@always_inline
fn forward_mlp_nano(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],
    weights: LayerWeights,
    hidden_size: Int,
    intermediate_size: Int,
    layer_idx: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin]
):
    var gate_ptr = scratch_ptr
    var up_ptr = scratch_ptr + intermediate_size
    var geglu_out_ptr = scratch_ptr + intermediate_size * 2

    vec_mat_mul(gate_ptr, x_ptr, weights.gate_proj.ptr, hidden_size, intermediate_size)
    # Gemma3n defaults: first 10 layers use 0.95 activation sparsity, rest dense.
    if layer_idx < 10:
        _apply_nano_activation_sparsity(gate_ptr, intermediate_size, 0.95)
    vec_mat_mul(up_ptr, x_ptr, weights.up_proj.ptr, hidden_size, intermediate_size)

    geglu(geglu_out_ptr, gate_ptr, up_ptr, intermediate_size)

    vec_mat_mul(out_ptr, geglu_out_ptr, weights.down_proj.ptr, intermediate_size, hidden_size)

@always_inline
fn forward_layer(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],
    weights: LayerWeights,
    pos: Int,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    intermediate_size: Int,
    freqs_cos_ptr: UnsafePointer[Float32, MutExternalOrigin],
    freqs_sin_ptr: UnsafePointer[Float32, MutExternalOrigin],
    kv_cache_k_ptr: UnsafePointer[Float32, MutExternalOrigin],
    kv_cache_v_ptr: UnsafePointer[Float32, MutExternalOrigin],
    max_seq_len: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin]
):
    var norm_x_ptr = scratch_ptr
    var attn_out_ptr = scratch_ptr + hidden_size
    var attn_scratch_ptr = scratch_ptr + hidden_size * 2
    
    rms_norm(norm_x_ptr, x_ptr, weights.input_layernorm.ptr, hidden_size, 1e-6)
    
    forward_attention(
        attn_out_ptr, norm_x_ptr, weights, pos, hidden_size, num_heads, num_kv_heads,
        head_dim, freqs_cos_ptr, freqs_sin_ptr, kv_cache_k_ptr, kv_cache_v_ptr, max_seq_len, attn_scratch_ptr
    )
    
    var post_attn_out_ptr = scratch_ptr + hidden_size * 2
    rms_norm(post_attn_out_ptr, attn_out_ptr, weights.post_attention_layernorm.ptr, hidden_size, 1e-6)
    
    var residual_ptr = scratch_ptr + hidden_size * 3
    for i in range(hidden_size):
        residual_ptr.store(i, x_ptr.load(i) + post_attn_out_ptr.load(i))
        
    var norm_residual_ptr = scratch_ptr + hidden_size * 4
    rms_norm(norm_residual_ptr, residual_ptr, weights.pre_feedforward_layernorm.ptr, hidden_size, 1e-6)
    
    var mlp_out_ptr = scratch_ptr + hidden_size * 5
    var mlp_scratch_ptr = scratch_ptr + hidden_size * 6
    forward_mlp(mlp_out_ptr, norm_residual_ptr, weights, hidden_size, intermediate_size, mlp_scratch_ptr)
    
    var post_mlp_out_ptr = scratch_ptr + hidden_size * 7
    rms_norm(post_mlp_out_ptr, mlp_out_ptr, weights.post_feedforward_layernorm.ptr, hidden_size, 1e-6)
    
    for i in range(hidden_size):
        out_ptr.store(i, residual_ptr.load(i) + post_mlp_out_ptr.load(i))


@always_inline
fn forward_step(
    out_logits_ptr: UnsafePointer[Float32, MutExternalOrigin], # [vocab_size]
    token_id: Int,
    pos: Int,
    model: ModelWeights,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    intermediate_size: Int,
    vocab_size: Int,
    freqs_cos_ptr: UnsafePointer[Float32, MutExternalOrigin], # [max_seq_len, head_dim]
    freqs_sin_ptr: UnsafePointer[Float32, MutExternalOrigin], # [max_seq_len, head_dim]
    kv_cache_k_ptr: UnsafePointer[Float32, MutExternalOrigin], # [num_layers, max_seq_len, num_kv_heads, head_dim]
    kv_cache_v_ptr: UnsafePointer[Float32, MutExternalOrigin], # [num_layers, max_seq_len, num_kv_heads, head_dim]
    max_seq_len: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin] # temp memory
):
    var num_layers = len(model.layers)
    var current_state = scratch_ptr
    var next_state = scratch_ptr + hidden_size
    var layer_scratch = scratch_ptr + hidden_size * 2
    
    var emb_scale = sqrt(Float32(hidden_size))
    # Initial embedding
    model.get_embedding(token_id, current_state)
    for i in range(hidden_size):
        current_state.store(i, current_state.load(i) * emb_scale)
    
    # Pass through layers
    for l in range(num_layers):
        var layer_kv_k_ptr = kv_cache_k_ptr + l * max_seq_len * num_kv_heads * head_dim
        var layer_kv_v_ptr = kv_cache_v_ptr + l * max_seq_len * num_kv_heads * head_dim
        
        var token_freqs_cos_ptr = freqs_cos_ptr + pos * head_dim
        var token_freqs_sin_ptr = freqs_sin_ptr + pos * head_dim
        
        forward_layer(
            next_state,
            current_state,
            model.layers[l],
            pos,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            token_freqs_cos_ptr,
            token_freqs_sin_ptr,
            layer_kv_k_ptr,
            layer_kv_v_ptr,
            max_seq_len,
            layer_scratch
        )
        
        # Swap states
        for i in range(hidden_size):
            current_state.store(i, next_state.load(i))
            
    # Final Norm for this token
    var norm_out = next_state # re-use next_state for norm out
    rms_norm(norm_out, current_state, model.norm.ptr, hidden_size, 1e-6)
    
    # Final LM Head projection
    vec_mat_mul(out_logits_ptr, norm_out, model.lm_head.ptr, hidden_size, vocab_size)


@always_inline
fn forward_sequence(
    out_emb_ptr: UnsafePointer[Float32, MutExternalOrigin], # [hidden_size]
    input_ids_ptr: UnsafePointer[Int32, MutExternalOrigin], # [seq_len]
    seq_len: Int,
    model: ModelWeights,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    intermediate_size: Int,
    freqs_cos_ptr: UnsafePointer[Float32, MutExternalOrigin], # [max_seq_len, head_dim]
    freqs_sin_ptr: UnsafePointer[Float32, MutExternalOrigin], # [max_seq_len, head_dim]
    kv_cache_k_ptr: UnsafePointer[Float32, MutExternalOrigin], # [num_layers, max_seq_len, num_kv_heads, head_dim]
    kv_cache_v_ptr: UnsafePointer[Float32, MutExternalOrigin], # [num_layers, max_seq_len, num_kv_heads, head_dim]
    max_seq_len: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin] # temp memory
):
    var num_layers = len(model.layers)
    var current_state = scratch_ptr
    var next_state = scratch_ptr + hidden_size
    var layer_scratch = scratch_ptr + hidden_size * 2
    
    # 1. Zero out embedding accumulator for mean pooling
    var emb_acc = layer_scratch + hidden_size * 10 # use memory far ahead
    for i in range(hidden_size):
        emb_acc.store(i, 0.0)

    # Process each token
    for t in range(seq_len):
        var token_id = Int(input_ids_ptr.load(t))
        
        var emb_scale = sqrt(Float32(hidden_size))
        # Initial embedding
        model.get_embedding(token_id, current_state)
        for i in range(hidden_size):
            current_state.store(i, current_state.load(i) * emb_scale)
        
        # Pass through layers
        for l in range(num_layers):
            var layer_kv_k_ptr = kv_cache_k_ptr + l * max_seq_len * num_kv_heads * head_dim
            var layer_kv_v_ptr = kv_cache_v_ptr + l * max_seq_len * num_kv_heads * head_dim
            
            var token_freqs_cos_ptr = freqs_cos_ptr + t * head_dim
            var token_freqs_sin_ptr = freqs_sin_ptr + t * head_dim
            
            forward_layer(
                next_state,
                current_state,
                model.layers[l],
                t,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                intermediate_size,
                token_freqs_cos_ptr,
                token_freqs_sin_ptr,
                layer_kv_k_ptr,
                layer_kv_v_ptr,
                max_seq_len,
                layer_scratch
            )
            
            # Swap states
            for i in range(hidden_size):
                current_state.store(i, next_state.load(i))
                
        # Final Norm for this token
        var norm_out = next_state # re-use next_state for norm out
        rms_norm(norm_out, current_state, model.norm.ptr, hidden_size, 1e-6)
        
        # Add to accumulator for mean pooling
        for i in range(hidden_size):
            emb_acc.store(i, emb_acc.load(i) + norm_out.load(i))
            
    # Calculate mean
    var scale = 1.0 / Float32(seq_len)
    for i in range(hidden_size):
        out_emb_ptr.store(i, emb_acc.load(i) * scale)

@always_inline
fn _rms_norm_nano_weighted(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],
    weight_ptr: UnsafePointer[Float32, MutExternalOrigin],
    size: Int,
    eps: Float32 = 1e-6,
):
    var sum_sq: Float32 = 0.0
    for i in range(size):
        var v = x_ptr.load(i)
        sum_sq += v * v
    var inv_rms = 1.0 / sqrt(sum_sq / Float32(size) + eps)
    for i in range(size):
        out_ptr.store(i, x_ptr.load(i) * inv_rms * weight_ptr.load(i))

@always_inline
fn _rms_norm_nano_unit(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],
    size: Int,
    eps: Float32 = 1e-6,
):
    var sum_sq: Float32 = 0.0
    for i in range(size):
        var v = x_ptr.load(i)
        sum_sq += v * v
    var inv_rms = 1.0 / sqrt(sum_sq / Float32(size) + eps)
    for i in range(size):
        out_ptr.store(i, x_ptr.load(i) * inv_rms)

@always_inline
fn forward_attention_nano(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],
    weights: LayerWeights,
    pos: Int,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    freqs_cos_ptr: UnsafePointer[Float32, MutExternalOrigin],
    freqs_sin_ptr: UnsafePointer[Float32, MutExternalOrigin],
    kv_cache_k_ptr: UnsafePointer[Float32, MutExternalOrigin],
    kv_cache_v_ptr: UnsafePointer[Float32, MutExternalOrigin],
    max_seq_len: Int,
    write_kv: Bool,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin],
):
    var q_size = num_heads * head_dim
    var kv_size = num_kv_heads * head_dim
    var q_ptr = scratch_ptr
    var k_ptr = scratch_ptr + q_size
    var v_ptr = scratch_ptr + q_size + kv_size

    vec_mat_mul(q_ptr, x_ptr, weights.q_proj.ptr, hidden_size, q_size)
    # Gemma3n attention uses weighted RMSNorm on q/k and unweighted RMSNorm on v.
    for h in range(num_heads):
        _rms_norm_nano_weighted(q_ptr + h * head_dim, q_ptr + h * head_dim, weights.q_norm.ptr, head_dim, 1e-6)

    for h in range(num_heads):
        rope_rotate(q_ptr + h * head_dim, freqs_cos_ptr, freqs_sin_ptr, head_dim)

    if write_kv:
        vec_mat_mul(k_ptr, x_ptr, weights.k_proj.ptr, hidden_size, kv_size)
        vec_mat_mul(v_ptr, x_ptr, weights.v_proj.ptr, hidden_size, kv_size)
        for h in range(num_kv_heads):
            _rms_norm_nano_weighted(k_ptr + h * head_dim, k_ptr + h * head_dim, weights.k_norm.ptr, head_dim, 1e-6)
            _rms_norm_nano_unit(v_ptr + h * head_dim, v_ptr + h * head_dim, head_dim, 1e-6)
            rope_rotate(k_ptr + h * head_dim, freqs_cos_ptr, freqs_sin_ptr, head_dim)

        var kv_offset = pos * kv_size
        for i in range(kv_size):
            kv_cache_k_ptr.store(kv_offset + i, k_ptr.load(i))
            kv_cache_v_ptr.store(kv_offset + i, v_ptr.load(i))

    var heads_per_kv = num_heads // num_kv_heads
    var attn_out_ptr = scratch_ptr + q_size + kv_size + kv_size

    # Gemma3n sets attention scaling to 1.0.
    for h in range(num_heads):
        var kv_h = h // heads_per_kv
        var q_head_ptr = q_ptr + h * head_dim
        var scores_ptr = attn_out_ptr + num_heads * head_dim

        for t in range(pos + 1):
            var k_head_ptr = kv_cache_k_ptr + t * kv_size + kv_h * head_dim
            var score: Float32 = 0.0
            for d in range(head_dim):
                score += q_head_ptr.load(d) * k_head_ptr.load(d)
            scores_ptr.store(t, score)

        softmax(scores_ptr, pos + 1)

        var out_head_ptr = attn_out_ptr + h * head_dim
        for d in range(head_dim):
            out_head_ptr.store(d, 0.0)

        for t in range(pos + 1):
            var v_head_ptr = kv_cache_v_ptr + t * kv_size + kv_h * head_dim
            var prob = scores_ptr.load(t)
            for d in range(head_dim):
                var acc = out_head_ptr.load(d)
                out_head_ptr.store(d, acc + prob * v_head_ptr.load(d))

    vec_mat_mul(out_ptr, attn_out_ptr, weights.o_proj.ptr, q_size, hidden_size)

@always_inline
fn forward_per_layer_mapping(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin], # [hidden_size] delta to add to non-active streams
    active_ptr: UnsafePointer[Float32, MutExternalOrigin], # active corrected prediction
    per_layer_input_ptr: UnsafePointer[Float32, MutExternalOrigin], # [per_layer_dim]
    weights: PerLayerMapWeights,
    hidden_size: Int,
    per_layer_dim: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin]
):
    var gate_out_ptr = scratch_ptr
    var proj_out_ptr = scratch_ptr + per_layer_dim

    # per_layer_input_gate + activation + multiply(per_layer_input)
    vec_mat_mul(gate_out_ptr, active_ptr, weights.gate.ptr, hidden_size, per_layer_dim)
    var sqrt_2: Float32 = 1.4142135623730951
    for i in range(per_layer_dim):
        var x = gate_out_ptr.load(i)
        var gelu_x = 0.5 * x * (1.0 + erf(x / sqrt_2))
        gate_out_ptr.store(i, gelu_x * per_layer_input_ptr.load(i))

    # per_layer_projection + post_per_layer_input_norm
    vec_mat_mul(proj_out_ptr, gate_out_ptr, weights.projection.ptr, per_layer_dim, hidden_size)
    _rms_norm_nano_weighted(out_ptr, proj_out_ptr, weights.norm.ptr, hidden_size, 1e-6)

@always_inline
fn forward_laurel(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    hidden_ptr: UnsafePointer[Float32, MutExternalOrigin],
    weights: LaurelWeights,
    hidden_size: Int,
    bottleneck_dim: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin]
):
    var down_ptr = scratch_ptr
    var up_ptr = scratch_ptr + bottleneck_dim
    var norm_up_ptr = up_ptr + hidden_size

    # laurel(hidden) = hidden + rms_norm(up(down(hidden)))
    vec_mat_mul(down_ptr, hidden_ptr, weights.down_proj.ptr, hidden_size, bottleneck_dim)
    vec_mat_mul(up_ptr, down_ptr, weights.up_proj.ptr, bottleneck_dim, hidden_size)
    _rms_norm_nano_weighted(norm_up_ptr, up_ptr, weights.norm.ptr, hidden_size, 1e-6)
    for i in range(hidden_size):
        out_ptr.store(i, hidden_ptr.load(i) + norm_up_ptr.load(i))

@always_inline
fn _compute_router_modalities(
    out_modalities_ptr: UnsafePointer[Float32, MutExternalOrigin], # [num_modalities]
    active_ptr: UnsafePointer[Float32, MutExternalOrigin], # [hidden_size]
    weights: AltUpWeights,
    hidden_size: Int,
    num_modalities: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin]
):
    var router_in_ptr = scratch_ptr
    _rms_norm_nano_weighted(router_in_ptr, active_ptr, weights.router_norm.ptr, hidden_size, 1e-6)

    var router_input_scale = 1.0 / Float32(hidden_size)
    for i in range(hidden_size):
        router_in_ptr.store(i, router_in_ptr.load(i) * router_input_scale)

    vec_mat_mul(out_modalities_ptr, router_in_ptr, weights.router.ptr, hidden_size, num_modalities)
    for m in range(num_modalities):
        out_modalities_ptr.store(m, tanh(out_modalities_ptr.load(m)))

@always_inline
fn forward_altup_predict(
    out_predictions_ptr: UnsafePointer[Float32, MutExternalOrigin], # [num_modalities, hidden_size]
    streams_ptr: UnsafePointer[Float32, MutExternalOrigin], # [num_modalities, hidden_size]
    weights: AltUpWeights,
    hidden_size: Int,
    num_modalities: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin]
):
    var modalities_ptr = scratch_ptr # [num_modalities]
    var coef_ptr = modalities_ptr + num_modalities # [num_modalities, num_modalities]
    var router_scratch_ptr = coef_ptr + num_modalities * num_modalities

    _compute_router_modalities(
        modalities_ptr,
        streams_ptr, # active idx = 0
        weights,
        hidden_size,
        num_modalities,
        router_scratch_ptr
    )

    # prediction_coefs layout follows JAX einsum convention: [router_modality, out_modality, in_modality]
    for in_m in range(num_modalities):
        for out_m in range(num_modalities):
            var coef: Float32 = 0.0
            for router_m in range(num_modalities):
                var idx = router_m * num_modalities * num_modalities + out_m * num_modalities + in_m
                coef += weights.prediction_coefs.ptr.load(idx) * modalities_ptr.load(router_m)
            coef_ptr.store(in_m * num_modalities + out_m, coef)

    for out_m in range(num_modalities):
        var out_base = out_predictions_ptr + out_m * hidden_size
        for d in range(hidden_size):
            var pred = streams_ptr.load(out_m * hidden_size + d)
            for in_m in range(num_modalities):
                pred += streams_ptr.load(in_m * hidden_size + d) * coef_ptr.load(in_m * num_modalities + out_m)
            out_base.store(d, pred)

@always_inline
fn forward_altup_correct(
    out_corrected_ptr: UnsafePointer[Float32, MutExternalOrigin], # [num_modalities, hidden_size]
    predictions_ptr: UnsafePointer[Float32, MutExternalOrigin], # [num_modalities, hidden_size]
    activated_ptr: UnsafePointer[Float32, MutExternalOrigin], # [hidden_size]
    weights: AltUpWeights,
    hidden_size: Int,
    num_modalities: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin]
):
    var modalities_ptr = scratch_ptr # [num_modalities]
    var corr_ptr = modalities_ptr + num_modalities # [num_modalities]
    var router_scratch_ptr = corr_ptr + num_modalities

    _compute_router_modalities(
        modalities_ptr,
        activated_ptr,
        weights,
        hidden_size,
        num_modalities,
        router_scratch_ptr
    )

    # correction_coefs layout: [router_modality, out_modality]
    for out_m in range(num_modalities):
        var coef: Float32 = 1.0
        for router_m in range(num_modalities):
            coef += weights.correction_coefs.ptr.load(router_m * num_modalities + out_m) * modalities_ptr.load(router_m)
        corr_ptr.store(out_m, coef)

    for d in range(hidden_size):
        var innovation = activated_ptr.load(d) - predictions_ptr.load(d) # active idx = 0
        for out_m in range(num_modalities):
            var base_idx = out_m * hidden_size + d
            out_corrected_ptr.store(base_idx, predictions_ptr.load(base_idx) + innovation * corr_ptr.load(out_m))

from collections import List

@always_inline
fn _vector_magnitude(
    vec_ptr: UnsafePointer[Float32, MutExternalOrigin],
    size: Int
) -> Float32:
    var sum_sq: Float32 = 0.0
    for i in range(size):
        var v = vec_ptr.load(i)
        sum_sq += v * v
    return sqrt(sum_sq / Float32(size) + 1e-6)

@always_inline
fn _prepare_altup_streams(
    out_streams_ptr: UnsafePointer[Float32, MutExternalOrigin], # [num_modalities, hidden_size]
    base_ptr: UnsafePointer[Float32, MutExternalOrigin], # [hidden_size]
    altup_projections: List[TensorInfo],
    hidden_size: Int,
    num_modalities: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin]
):
    # stream 0 = base stream
    for d in range(hidden_size):
        out_streams_ptr.store(d, base_ptr.load(d))

    var target_mag = _vector_magnitude(base_ptr, hidden_size)
    var proj_ptr = scratch_ptr

    # streams 1..N are projected and magnitude-matched
    for i in range(num_modalities - 1):
        vec_mat_mul(proj_ptr, base_ptr, altup_projections[i].ptr, hidden_size, hidden_size)
        var proj_mag = _vector_magnitude(proj_ptr, hidden_size)
        var safe_proj_mag = proj_mag
        if safe_proj_mag < 1e-6:
            safe_proj_mag = 1e-6
        var scale = target_mag / safe_proj_mag
        for d in range(hidden_size):
            out_streams_ptr.store((i + 1) * hidden_size + d, proj_ptr.load(d) * scale)

@always_inline
fn _build_token_per_layer_inputs(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin], # [num_layers, per_layer_dim]
    base_stream_ptr: UnsafePointer[Float32, MutExternalOrigin], # [hidden_size]
    token_id: Int,
    model: NanoModelWeights,
    num_layers: Int,
    hidden_size: Int,
    per_layer_dim: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin]
):
    var proj_ptr = scratch_ptr
    var proj_norm_ptr = scratch_ptr + per_layer_dim

    var projection_scale = 1.0 / sqrt(Float32(hidden_size))
    var per_layer_input_scale: Float32 = 0.7071067811865475
    var token_embed_scale = sqrt(Float32(per_layer_dim))
    var per_layer_table_layers = model.per_layer_embed.shape_1

    for l in range(num_layers):
        # Project base stream to per-layer input: W is [hidden_size, num_layers, per_layer_dim]
        for p in range(per_layer_dim):
            var acc: Float32 = 0.0
            for d in range(hidden_size):
                var w_idx = d * per_layer_table_layers * per_layer_dim + l * per_layer_dim + p
                acc += base_stream_ptr.load(d) * model.per_layer_projection.ptr.load(w_idx)
            proj_ptr.store(p, acc * projection_scale)

        _rms_norm_nano_weighted(proj_norm_ptr, proj_ptr, model.per_layer_norm.ptr, per_layer_dim, 1e-6)

        var embed_base = model.per_layer_embed.ptr + token_id * per_layer_table_layers * per_layer_dim + l * per_layer_dim
        var out_base = out_ptr + l * per_layer_dim
        for p in range(per_layer_dim):
            var token_embed = embed_base.load(p) * token_embed_scale
            out_base.store(p, (proj_norm_ptr.load(p) + token_embed) * per_layer_input_scale)

@always_inline
fn _collapse_altup_streams(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin], # [hidden_size]
    streams_ptr: UnsafePointer[Float32, MutExternalOrigin], # [num_modalities, hidden_size]
    altup_unembeds: List[TensorInfo],
    hidden_size: Int,
    num_modalities: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin]
):
    var tmp_ptr = scratch_ptr
    var acc_ptr = scratch_ptr + hidden_size

    var target_mag = _vector_magnitude(streams_ptr, hidden_size)
    for d in range(hidden_size):
        acc_ptr.store(d, streams_ptr.load(d))

    for m in range(1, num_modalities):
        var stream_ptr = streams_ptr + m * hidden_size
        vec_mat_mul(tmp_ptr, stream_ptr, altup_unembeds[m - 1].ptr, hidden_size, hidden_size)
        var new_mag = _vector_magnitude(tmp_ptr, hidden_size)
        var safe_new_mag = new_mag
        if safe_new_mag < 1e-6:
            safe_new_mag = 1e-6
        var scale = target_mag / safe_new_mag
        for d in range(hidden_size):
            acc_ptr.store(d, acc_ptr.load(d) + tmp_ptr.load(d) * scale)

    var inv_modalities = 1.0 / Float32(num_modalities)
    for d in range(hidden_size):
        out_ptr.store(d, acc_ptr.load(d) * inv_modalities)


@always_inline
fn forward_nano_layer(
    out_streams_ptr: UnsafePointer[Float32, MutExternalOrigin], # [num_modalities, hidden_size]
    in_streams_ptr: UnsafePointer[Float32, MutExternalOrigin], # [num_modalities, hidden_size]
    weights: NanoLayerWeights,
    layer_idx: Int,
    per_layer_input_ptr: UnsafePointer[Float32, MutExternalOrigin], # [per_layer_dim]
    pos: Int,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    intermediate_size: Int,
    per_layer_dim: Int,
    freqs_cos_ptr: UnsafePointer[Float32, MutExternalOrigin],
    freqs_sin_ptr: UnsafePointer[Float32, MutExternalOrigin],
    kv_cache_k_ptr: UnsafePointer[Float32, MutExternalOrigin],
    kv_cache_v_ptr: UnsafePointer[Float32, MutExternalOrigin],
    max_seq_len: Int,
    num_modalities: Int,
    write_kv: Bool,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin]
):
    var predictions_ptr = scratch_ptr                                  # 0..4h
    var corrected_ptr = predictions_ptr + num_modalities * hidden_size # 4h..8h
    var active_ptr = corrected_ptr + num_modalities * hidden_size      # 8h
    var active_norm_ptr = active_ptr + hidden_size                     # 9h
    var laurel_ptr = active_norm_ptr + hidden_size                     # 10h
    var attn_ptr = laurel_ptr + hidden_size                            # 11h
    var attn_norm_ptr = attn_ptr + hidden_size                         # 12h
    var attn_laurel_ptr = attn_norm_ptr + hidden_size                  # 13h
    var ffw_norm_in_ptr = attn_laurel_ptr + hidden_size                # 14h
    var ffw_ptr = ffw_norm_in_ptr + hidden_size                        # 15h
    var ffw_norm_ptr = ffw_ptr + hidden_size                           # 16h
    var activated_ptr = ffw_norm_ptr + hidden_size                     # 17h
    var first_prediction_ptr = activated_ptr + hidden_size             # 18h
    var delta_ptr = first_prediction_ptr + hidden_size                 # 19h
    var altup_scratch_ptr = delta_ptr + hidden_size                    # 20h
    var attn_scratch_ptr = scratch_ptr + hidden_size * 24
    var laurel_scratch_ptr = scratch_ptr + hidden_size * 40
    var ffw_scratch_ptr = scratch_ptr + hidden_size * 44
    var plm_scratch_ptr = scratch_ptr + hidden_size * 48

    forward_altup_predict(
        predictions_ptr,
        in_streams_ptr,
        weights.altup,
        hidden_size,
        num_modalities,
        altup_scratch_ptr
    )

    # Active stream (index 0) runs attention/MLP path
    for i in range(hidden_size):
        active_ptr.store(i, predictions_ptr.load(i))

    _rms_norm_nano_weighted(active_norm_ptr, active_ptr, weights.base.input_layernorm.ptr, hidden_size, 1e-6)
    forward_laurel(
        laurel_ptr,
        active_norm_ptr,
        weights.laurel,
        hidden_size,
        weights.laurel.down_proj.shape_0,
        laurel_scratch_ptr
    )

    forward_attention_nano(
        attn_ptr, active_norm_ptr, weights.base, pos, hidden_size, num_heads, num_kv_heads,
        head_dim, freqs_cos_ptr, freqs_sin_ptr, kv_cache_k_ptr, kv_cache_v_ptr, max_seq_len, write_kv, attn_scratch_ptr
    )

    _rms_norm_nano_weighted(attn_norm_ptr, attn_ptr, weights.base.post_attention_layernorm.ptr, hidden_size, 1e-6)
    var inv_sqrt2: Float32 = 0.7071067811865475
    for i in range(hidden_size):
        attn_laurel_ptr.store(i, (active_ptr.load(i) + attn_norm_ptr.load(i) + laurel_ptr.load(i)) * inv_sqrt2)

    _rms_norm_nano_weighted(ffw_norm_in_ptr, attn_laurel_ptr, weights.base.pre_feedforward_layernorm.ptr, hidden_size, 1e-6)
    forward_mlp_nano(ffw_ptr, ffw_norm_in_ptr, weights.base, hidden_size, intermediate_size, layer_idx, ffw_scratch_ptr)
    _rms_norm_nano_weighted(ffw_norm_ptr, ffw_ptr, weights.base.post_feedforward_layernorm.ptr, hidden_size, 1e-6)
    for i in range(hidden_size):
        activated_ptr.store(i, attn_laurel_ptr.load(i) + ffw_norm_ptr.load(i))

    forward_altup_correct(
        corrected_ptr,
        predictions_ptr,
        activated_ptr,
        weights.altup,
        hidden_size,
        num_modalities,
        altup_scratch_ptr
    )

    # Active corrected prediction is scaled by correct_output_scale
    for i in range(hidden_size):
        first_prediction_ptr.store(i, corrected_ptr.load(i) * weights.altup.output_scale.ptr.load(i))

    forward_per_layer_mapping(
        delta_ptr,
        first_prediction_ptr,
        per_layer_input_ptr,
        weights.per_layer_map,
        hidden_size,
        per_layer_dim,
        plm_scratch_ptr
    )

    # stream 0 unchanged; non-active streams receive per-layer delta
    for i in range(hidden_size):
        out_streams_ptr.store(i, corrected_ptr.load(i))
    for m in range(1, num_modalities):
        for i in range(hidden_size):
            var idx = m * hidden_size + i
            out_streams_ptr.store(idx, corrected_ptr.load(idx) + delta_ptr.load(i))

@always_inline
fn _forward_nano_token_hidden(
    out_hidden_ptr: UnsafePointer[Float32, MutExternalOrigin], # [hidden_size]
    token_id: Int,
    pos: Int,
    model: NanoModelWeights,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    intermediate_size: Int,
    per_layer_dim: Int,
    freqs_cos_ptr: UnsafePointer[Float32, MutExternalOrigin],
    freqs_sin_ptr: UnsafePointer[Float32, MutExternalOrigin],
    kv_cache_k_ptr: UnsafePointer[Float32, MutExternalOrigin],
    kv_cache_v_ptr: UnsafePointer[Float32, MutExternalOrigin],
    max_seq_len: Int,
    num_modalities: Int,
    kv_share_start: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin]
):
    var num_layers = len(model.layers)
    var current_streams_ptr = scratch_ptr
    var next_streams_ptr = current_streams_ptr + num_modalities * hidden_size
    var per_layer_inputs_ptr = next_streams_ptr + num_modalities * hidden_size
    var layer_scratch_ptr = per_layer_inputs_ptr + num_layers * per_layer_dim
    var collapse_scratch_ptr = layer_scratch_ptr + hidden_size * 72
    var stream_init_scratch_ptr = layer_scratch_ptr + hidden_size * 68

    # Base token embedding (scaled)
    model.get_embedding(token_id, current_streams_ptr)
    var emb_scale = sqrt(Float32(hidden_size))
    for i in range(hidden_size):
        current_streams_ptr.store(i, current_streams_ptr.load(i) * emb_scale)

    _prepare_altup_streams(
        current_streams_ptr,
        current_streams_ptr,
        model.altup_projections,
        hidden_size,
        num_modalities,
        stream_init_scratch_ptr
    )

    _build_token_per_layer_inputs(
        per_layer_inputs_ptr,
        current_streams_ptr,
        token_id,
        model,
        num_layers,
        hidden_size,
        per_layer_dim,
        layer_scratch_ptr
    )

    # Gemma3n default attention pattern: every 5th layer is full attention, others are sliding.
    var last_full_kv_layer = kv_share_start - 1
    while last_full_kv_layer >= 0 and ((last_full_kv_layer + 1) % 5) != 0:
        last_full_kv_layer -= 1
    var last_sliding_kv_layer = kv_share_start - 1
    while last_sliding_kv_layer >= 0 and ((last_sliding_kv_layer + 1) % 5) == 0:
        last_sliding_kv_layer -= 1

    for l in range(num_layers):
        var kv_layer_idx = l
        var write_kv = True
        if kv_share_start < num_layers and l >= kv_share_start:
            write_kv = False
            if ((l + 1) % 5) == 0 and last_full_kv_layer >= 0:
                kv_layer_idx = last_full_kv_layer
            elif last_sliding_kv_layer >= 0:
                kv_layer_idx = last_sliding_kv_layer
            else:
                kv_layer_idx = kv_share_start - 1

        var layer_kv_k_ptr = kv_cache_k_ptr + kv_layer_idx * max_seq_len * num_kv_heads * head_dim
        var layer_kv_v_ptr = kv_cache_v_ptr + kv_layer_idx * max_seq_len * num_kv_heads * head_dim
        var token_freqs_cos_ptr = freqs_cos_ptr + pos * head_dim
        var token_freqs_sin_ptr = freqs_sin_ptr + pos * head_dim
        var layer_per_input_ptr = per_layer_inputs_ptr + l * per_layer_dim

        forward_nano_layer(
            next_streams_ptr,
            current_streams_ptr,
            model.layers[l],
            l,
            layer_per_input_ptr,
            pos,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            per_layer_dim,
            token_freqs_cos_ptr,
            token_freqs_sin_ptr,
            layer_kv_k_ptr,
            layer_kv_v_ptr,
            max_seq_len,
            num_modalities,
            write_kv,
            layer_scratch_ptr
        )

        for i in range(num_modalities * hidden_size):
            current_streams_ptr.store(i, next_streams_ptr.load(i))

    _collapse_altup_streams(
        out_hidden_ptr,
        current_streams_ptr,
        model.altup_unembeds,
        hidden_size,
        num_modalities,
        collapse_scratch_ptr
    )
    _rms_norm_nano_weighted(out_hidden_ptr, out_hidden_ptr, model.norm.ptr, hidden_size, 1e-6)

@always_inline
fn forward_nano_step(
    out_logits_ptr: UnsafePointer[Float32, MutExternalOrigin], # [vocab_size]
    token_id: Int,
    pos: Int,
    model: NanoModelWeights,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    intermediate_size: Int,
    per_layer_dim: Int,
    vocab_size: Int,
    freqs_cos_ptr: UnsafePointer[Float32, MutExternalOrigin], # [max_seq_len, head_dim]
    freqs_sin_ptr: UnsafePointer[Float32, MutExternalOrigin], # [max_seq_len, head_dim]
    kv_cache_k_ptr: UnsafePointer[Float32, MutExternalOrigin], # [num_layers, max_seq_len, num_kv_heads, head_dim]
    kv_cache_v_ptr: UnsafePointer[Float32, MutExternalOrigin], # [num_layers, max_seq_len, num_kv_heads, head_dim]
    max_seq_len: Int,
    kv_share_start: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin] # temp memory
):
    var num_modalities = model.layers[0].altup.router.shape_0
    var hidden_ptr = scratch_ptr
    var token_scratch_ptr = scratch_ptr + hidden_size

    _forward_nano_token_hidden(
        hidden_ptr,
        token_id,
        pos,
        model,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        per_layer_dim,
        freqs_cos_ptr,
        freqs_sin_ptr,
        kv_cache_k_ptr,
        kv_cache_v_ptr,
        max_seq_len,
        num_modalities,
        kv_share_start,
        token_scratch_ptr
    )

    vec_mat_mul(out_logits_ptr, hidden_ptr, model.lm_head.ptr, hidden_size, vocab_size)

@always_inline
fn forward_nano_sequence(
    out_emb_ptr: UnsafePointer[Float32, MutExternalOrigin], # [hidden_size]
    input_ids_ptr: UnsafePointer[Int32, MutExternalOrigin], # [seq_len]
    seq_len: Int,
    model: NanoModelWeights,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    intermediate_size: Int,
    per_layer_dim: Int,
    freqs_cos_ptr: UnsafePointer[Float32, MutExternalOrigin], # [max_seq_len, head_dim]
    freqs_sin_ptr: UnsafePointer[Float32, MutExternalOrigin], # [max_seq_len, head_dim]
    kv_cache_k_ptr: UnsafePointer[Float32, MutExternalOrigin], # [num_layers, max_seq_len, num_kv_heads, head_dim]
    kv_cache_v_ptr: UnsafePointer[Float32, MutExternalOrigin], # [num_layers, max_seq_len, num_kv_heads, head_dim]
    max_seq_len: Int,
    kv_share_start: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin] # temp memory
):
    var emb_acc = scratch_ptr
    var token_hidden_ptr = scratch_ptr + hidden_size
    var token_scratch_ptr = scratch_ptr + hidden_size * 2

    for i in range(hidden_size):
        emb_acc.store(i, 0.0)

    for t in range(seq_len):
        var token_id = Int(input_ids_ptr.load(t))

        var num_modalities = model.layers[0].altup.router.shape_0
        _forward_nano_token_hidden(
            token_hidden_ptr,
            token_id,
            t,
            model,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            per_layer_dim,
            freqs_cos_ptr,
            freqs_sin_ptr,
            kv_cache_k_ptr,
            kv_cache_v_ptr,
            max_seq_len,
            num_modalities,
            kv_share_start,
            token_scratch_ptr
        )

        for i in range(hidden_size):
            emb_acc.store(i, emb_acc.load(i) + token_hidden_ptr.load(i))

    var scale = 1.0 / Float32(seq_len)
    for i in range(hidden_size):
        out_emb_ptr.store(i, emb_acc.load(i) * scale)
