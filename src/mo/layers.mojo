from memory import UnsafePointer
from math import sqrt
from model import LayerWeights, ModelWeights
from ops import vec_mat_mul, rope_rotate, softmax, rms_norm, geglu

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
