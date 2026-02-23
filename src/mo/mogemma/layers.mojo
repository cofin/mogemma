from memory import UnsafePointer
from math import sqrt
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
    
    # 1. Input RMSNorm
    rms_norm(norm_x_ptr, x_ptr, weights.input_layernorm.ptr, hidden_size, 1e-6)
    
    # 2. Attention
    forward_attention(
        attn_out_ptr,
        norm_x_ptr,
        weights,
        pos,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        freqs_cos_ptr,
        freqs_sin_ptr,
        kv_cache_k_ptr,
        kv_cache_v_ptr,
        max_seq_len,
        attn_scratch_ptr
    )
    
    # 3. Residual connection
    var residual_ptr = scratch_ptr + hidden_size * 2 # re-use memory
    for i in range(hidden_size):
        residual_ptr.store(i, x_ptr.load(i) + attn_out_ptr.load(i))
        
    # 4. Post-attention RMSNorm
    var norm_residual_ptr = scratch_ptr + hidden_size * 3
    rms_norm(norm_residual_ptr, residual_ptr, weights.post_attention_layernorm.ptr, hidden_size, 1e-6)
    
    # 5. MLP
    var mlp_out_ptr = scratch_ptr + hidden_size * 4
    var mlp_scratch_ptr = scratch_ptr + hidden_size * 5
    
    forward_mlp(
        mlp_out_ptr,
        norm_residual_ptr,
        weights,
        hidden_size,
        intermediate_size,
        mlp_scratch_ptr
    )
    
    # 6. Residual connection
    for i in range(hidden_size):
        out_ptr.store(i, residual_ptr.load(i) + mlp_out_ptr.load(i))


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
fn forward_per_layer_mapping(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin], # Unnormalized residual
    norm_x_ptr: UnsafePointer[Float32, MutExternalOrigin], # Normalized input
    layer_idx: Int,
    token_id: Int,
    per_layer_embed_ptr: UnsafePointer[Float32, MutExternalOrigin],
    weights: PerLayerMapWeights,
    hidden_size: Int,
    per_layer_dim: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin]
):
    var gate_out_ptr = scratch_ptr
    var proj_out_ptr = scratch_ptr + per_layer_dim
    
    # 1. Gate input
    vec_mat_mul(gate_out_ptr, norm_x_ptr, weights.gate.ptr, hidden_size, per_layer_dim)
    
    # 2. Lookup per-layer embedding
    # Shape is [vocab_size, num_layers, per_layer_dim]. We need layer_idx for this token.
    # The stride is (num_layers * per_layer_dim) per token, and (per_layer_dim) per layer.
    # In standard Gemma 3 Nano, num_layers = 30, per_layer_dim = 256.
    # (Checking shape from convert.py: (262144, 30, 256))
    var embed_offset = token_id * 30 * per_layer_dim + layer_idx * per_layer_dim
    var layer_embed_ptr = per_layer_embed_ptr + embed_offset
    
    # Element-wise multiply gate with per-layer embedding
    for i in range(per_layer_dim):
        gate_out_ptr.store(i, gate_out_ptr.load(i) * layer_embed_ptr.load(i))
        
    # 3. Project back to hidden_size
    vec_mat_mul(proj_out_ptr, gate_out_ptr, weights.projection.ptr, per_layer_dim, hidden_size)
    
    # 4. Add residual
    # Output is unnormalized residual
    for i in range(hidden_size):
        out_ptr.store(i, x_ptr.load(i) + proj_out_ptr.load(i))

@always_inline
fn forward_laurel(
    out_res_ptr: UnsafePointer[Float32, MutExternalOrigin], # Output unnormalized residual
    out_norm_ptr: UnsafePointer[Float32, MutExternalOrigin], # Output normalized residual
    res_ptr: UnsafePointer[Float32, MutExternalOrigin], # Input unnormalized residual
    norm_res_ptr: UnsafePointer[Float32, MutExternalOrigin], # Input normalized residual
    weights: LaurelWeights,
    hidden_size: Int,
    bottleneck_dim: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin]
):
    var down_ptr = scratch_ptr
    var up_ptr = scratch_ptr + bottleneck_dim
    
    # 1. Down project
    vec_mat_mul(down_ptr, norm_res_ptr, weights.down_proj.ptr, hidden_size, bottleneck_dim)
    
    # 2. Up project
    vec_mat_mul(up_ptr, down_ptr, weights.up_proj.ptr, bottleneck_dim, hidden_size)
    
    # 3. Add residual
    for i in range(hidden_size):
        out_res_ptr.store(i, res_ptr.load(i) + up_ptr.load(i))
        
    # 4. Norm
    rms_norm(out_norm_ptr, out_res_ptr, weights.norm.ptr, hidden_size, 1e-6)

@always_inline
fn forward_altup_predict(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    variants_ptr: UnsafePointer[Float32, MutExternalOrigin],
    router_probs_ptr: UnsafePointer[Float32, MutExternalOrigin],
    weights: AltUpWeights,
    hidden_size: Int,
    num_modalities: Int
):
    # weights.prediction_coefs is shape (num_modalities, hidden_size)
    # variants_ptr is (num_modalities, hidden_size)
    for d in range(hidden_size):
        var pred: Float32 = 0.0
        for m in range(num_modalities):
            var prob = router_probs_ptr.load(m)
            var coef = weights.prediction_coefs.ptr.load(m * hidden_size + d)
            var val = variants_ptr.load(m * hidden_size + d)
            pred += prob * coef * val
        out_ptr.store(d, pred)

@always_inline
fn forward_altup_correct(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    predicted_ptr: UnsafePointer[Float32, MutExternalOrigin],
    weights: AltUpWeights,
    hidden_size: Int
):
    # weights.correction_coefs is (hidden_size,)
    # weights.output_scale is (hidden_size,)
    for d in range(hidden_size):
        var corrected = predicted_ptr.load(d) + weights.correction_coefs.ptr.load(d)
        out_ptr.store(d, corrected * weights.output_scale.ptr.load(d))

from collections import List

@always_inline
fn forward_altup(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    res_ptr: UnsafePointer[Float32, MutExternalOrigin], # Unnormalized residual
    norm_res_ptr: UnsafePointer[Float32, MutExternalOrigin], # Normalized input
    altup_projections: List[TensorInfo],
    altup_unembeds: List[TensorInfo],
    weights: AltUpWeights,
    hidden_size: Int,
    num_modalities: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin]
):
    # 1. Project variants. Variant 0 is identity (norm_res_ptr). Variants 1..3 use projections.
    var variants_ptr = scratch_ptr # [num_modalities, hidden_size]
    for d in range(hidden_size):
        variants_ptr.store(d, norm_res_ptr.load(d))
        
    for i in range(num_modalities - 1):
        vec_mat_mul(variants_ptr + (i + 1) * hidden_size, norm_res_ptr, altup_projections[i].ptr, hidden_size, hidden_size)
        
    # 2. Router
    var router_norm_ptr = scratch_ptr + num_modalities * hidden_size
    rms_norm(router_norm_ptr, norm_res_ptr, weights.router_norm.ptr, hidden_size, 1e-6)
    
    var router_logits_ptr = scratch_ptr + num_modalities * hidden_size + hidden_size
    vec_mat_mul(router_logits_ptr, router_norm_ptr, weights.router.ptr, hidden_size, num_modalities)
    
    softmax(router_logits_ptr, num_modalities)
    
    # 3. Predict & Unembed
    var predicted_ptr = scratch_ptr + num_modalities * hidden_size + hidden_size + num_modalities
    var unembed_scratch_ptr = predicted_ptr + num_modalities * hidden_size
    
    # The variants must be unembeded. 
    # For modality 0, it's just the original space.
    # For modalities 1..3, we unembed.
    # We can apply predict per-modality, then unembed, then sum.
    var combined_ptr = unembed_scratch_ptr + hidden_size
    for d in range(hidden_size):
        combined_ptr.store(d, 0.0)
        
    for m in range(num_modalities):
        var prob = router_logits_ptr.load(m)
        var variant_base = variants_ptr + m * hidden_size
        var m_predicted_ptr = predicted_ptr + m * hidden_size
        
        for d in range(hidden_size):
            var coef = weights.prediction_coefs.ptr.load(m * hidden_size + d)
            var val = variant_base.load(d)
            m_predicted_ptr.store(d, prob * coef * val)
            
        if m == 0:
            for d in range(hidden_size):
                combined_ptr.store(d, combined_ptr.load(d) + m_predicted_ptr.load(d))
        else:
            # Unembed
            vec_mat_mul(unembed_scratch_ptr, m_predicted_ptr, altup_unembeds[m - 1].ptr, hidden_size, hidden_size)
            for d in range(hidden_size):
                combined_ptr.store(d, combined_ptr.load(d) + unembed_scratch_ptr.load(d))
                
    # 4. Correct and Scale
    var altup_out_ptr = unembed_scratch_ptr + hidden_size * 2
    forward_altup_correct(altup_out_ptr, combined_ptr, weights, hidden_size)
    
    # 5. Add to residual
    for d in range(hidden_size):
        out_ptr.store(d, res_ptr.load(d) + altup_out_ptr.load(d))


@always_inline
fn forward_nano_layer(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],
    weights: NanoLayerWeights,
    layer_idx: Int,
    token_id: Int,
    pos: Int,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    intermediate_size: Int,
    per_layer_dim: Int,
    bottleneck_dim: Int,
    freqs_cos_ptr: UnsafePointer[Float32, MutExternalOrigin],
    freqs_sin_ptr: UnsafePointer[Float32, MutExternalOrigin],
    kv_cache_k_ptr: UnsafePointer[Float32, MutExternalOrigin],
    kv_cache_v_ptr: UnsafePointer[Float32, MutExternalOrigin],
    max_seq_len: Int,
    per_layer_embed_ptr: UnsafePointer[Float32, MutExternalOrigin],
    altup_projections: List[TensorInfo],
    altup_unembeds: List[TensorInfo],
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin]
):
    var norm_x_ptr = scratch_ptr
    var attn_out_ptr = scratch_ptr + hidden_size
    var attn_scratch_ptr = scratch_ptr + hidden_size * 2
    
    # 1. Pre-Attention Norm
    rms_norm(norm_x_ptr, x_ptr, weights.base.input_layernorm.ptr, hidden_size, 1e-6)
    
    # 2. Attention
    forward_attention(
        attn_out_ptr,
        norm_x_ptr,
        weights.base,
        pos,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        freqs_cos_ptr,
        freqs_sin_ptr,
        kv_cache_k_ptr,
        kv_cache_v_ptr,
        max_seq_len,
        attn_scratch_ptr
    )
    
    # 3. Residual 1
    var res1_ptr = scratch_ptr + hidden_size * 2
    for i in range(hidden_size):
        res1_ptr.store(i, x_ptr.load(i) + attn_out_ptr.load(i))
        
    # 4. Post-Attention Norm
    var norm_res1_ptr = scratch_ptr + hidden_size * 3
    rms_norm(norm_res1_ptr, res1_ptr, weights.base.post_attention_layernorm.ptr, hidden_size, 1e-6)
    
    # 5. Per-Layer Mapping
    var plm_out_ptr = scratch_ptr + hidden_size * 4
    var plm_scratch_ptr = scratch_ptr + hidden_size * 5
    forward_per_layer_mapping(
        plm_out_ptr,
        res1_ptr,
        norm_res1_ptr,
        layer_idx,
        token_id,
        per_layer_embed_ptr,
        weights.per_layer_map,
        hidden_size,
        per_layer_dim,
        plm_scratch_ptr
    )
    
    # 6. Pre-FFW Norm
    var norm_plm_ptr = scratch_ptr + hidden_size * 5
    rms_norm(norm_plm_ptr, plm_out_ptr, weights.base.pre_feedforward_layernorm.ptr, hidden_size, 1e-6)
    
    # 7. MLP
    var mlp_out_ptr = scratch_ptr + hidden_size * 6
    var mlp_scratch_ptr = scratch_ptr + hidden_size * 7
    forward_mlp(
        mlp_out_ptr,
        norm_plm_ptr,
        weights.base,
        hidden_size,
        intermediate_size,
        mlp_scratch_ptr
    )
    
    # 8. Residual 2
    var res2_ptr = scratch_ptr + hidden_size * 7
    for i in range(hidden_size):
        res2_ptr.store(i, plm_out_ptr.load(i) + mlp_out_ptr.load(i))
        
    # 9. Post-FFW Norm
    var norm_res2_ptr = scratch_ptr + hidden_size * 8
    rms_norm(norm_res2_ptr, res2_ptr, weights.base.post_feedforward_layernorm.ptr, hidden_size, 1e-6)
    
    # 10. Laurel
    var res3_ptr = scratch_ptr + hidden_size * 9
    var norm_res3_ptr = scratch_ptr + hidden_size * 10
    var laurel_scratch_ptr = scratch_ptr + hidden_size * 11
    forward_laurel(
        res3_ptr,
        norm_res3_ptr,
        res2_ptr,
        norm_res2_ptr,
        weights.laurel,
        hidden_size,
        bottleneck_dim,
        laurel_scratch_ptr
    )
    
    # 11. AltUp
    var altup_scratch_ptr = scratch_ptr + hidden_size * 11
    forward_altup(
        out_ptr,
        res3_ptr,
        norm_res3_ptr,
        altup_projections,
        altup_unembeds,
        weights.altup,
        hidden_size,
        4,
        altup_scratch_ptr
    )


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
    bottleneck_dim: Int,
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
        
        forward_nano_layer(
            next_state,
            current_state,
            model.layers[l],
            l,
            token_id,
            pos,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            per_layer_dim,
            bottleneck_dim,
            token_freqs_cos_ptr,
            token_freqs_sin_ptr,
            layer_kv_k_ptr,
            layer_kv_v_ptr,
            max_seq_len,
            model.per_layer_embed.ptr,
            model.altup_projections,
            model.altup_unembeds,
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
    bottleneck_dim: Int,
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
    var emb_acc = layer_scratch + hidden_size * 20 # generous offset to avoid clash
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
            
            forward_nano_layer(
                next_state,
                current_state,
                model.layers[l],
                l,
                token_id,
                t,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                intermediate_size,
                per_layer_dim,
                bottleneck_dim,
                token_freqs_cos_ptr,
                token_freqs_sin_ptr,
                layer_kv_k_ptr,
                layer_kv_v_ptr,
                max_seq_len,
                model.per_layer_embed.ptr,
                model.altup_projections,
                model.altup_unembeds,
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
