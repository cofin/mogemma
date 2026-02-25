import numpy as np
import math

def geglu(gate, up):
    # Apply erf element-wise
    erf_vec = np.vectorize(math.erf)
    return 0.5 * gate * (1.0 + erf_vec(gate / np.sqrt(2.0))) * up

def rms_norm(x, weight, eps=1e-6):
    return x / np.sqrt(np.mean(x**2) + eps) * weight

def rope_rotate(x, cos, sin):
    # x is (D,), cos, sin are (D//2,)
    d2 = len(x) // 2
    x1, x2 = x[:d2], x[d2:]
    out = np.zeros_like(x)
    out[:d2] = x1 * cos - x2 * sin
    out[d2:] = x2 * cos + x1 * sin
    return out

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def forward_attention(x, pos, q_proj, k_proj, v_proj, o_proj, freqs_cos, freqs_sin, kv_cache_k, kv_cache_v, num_heads, num_kv_heads, head_dim):
    # Q, K, V Projections
    q = np.dot(q_proj, x) # (num_heads * head_dim,)
    k = np.dot(k_proj, x) # (num_kv_heads * head_dim,)
    v = np.dot(v_proj, x) # (num_kv_heads * head_dim,)
    
    # RoPE
    for h in range(num_heads):
        start = h * head_dim
        end = start + head_dim
        q[start:end] = rope_rotate(q[start:end], freqs_cos, freqs_sin)
    for h in range(num_kv_heads):
        start = h * head_dim
        end = start + head_dim
        k[start:end] = rope_rotate(k[start:end], freqs_cos, freqs_sin)

    # Update cache
    kv_cache_k[pos] = k
    kv_cache_v[pos] = v

    # Attention
    heads_per_kv = num_heads // num_kv_heads
    scale = 1.0 / np.sqrt(head_dim)
    
    attn_out = np.zeros_like(q)
    for h in range(num_heads):
        kv_h = h // heads_per_kv
        q_h = q[h * head_dim : (h + 1) * head_dim]
        
        scores = np.zeros(pos + 1, dtype=np.float32)
        for t in range(pos + 1):
            k_t_h = kv_cache_k[t, kv_h * head_dim : (kv_h + 1) * head_dim]
            scores[t] = np.dot(q_h, k_t_h) * scale
            
        probs = softmax(scores)
        
        out_h = np.zeros(head_dim, dtype=np.float32)
        for t in range(pos + 1):
            v_t_h = kv_cache_v[t, kv_h * head_dim : (kv_h + 1) * head_dim]
            out_h += probs[t] * v_t_h
            
        attn_out[h * head_dim : (h + 1) * head_dim] = out_h

    # Output projection
    return np.dot(o_proj, attn_out)

def forward_layer(x, pos, w, freqs_cos, freqs_sin, kv_cache_k, kv_cache_v, num_heads, num_kv_heads, head_dim, intermediate_size):
    # 1. Input RMSNorm
    norm_x = rms_norm(x, w['input_layernorm'])
    
    # 2. Attention
    attn_out = forward_attention(
        norm_x, pos, w['q_proj'], w['k_proj'], w['v_proj'], w['o_proj'],
        freqs_cos, freqs_sin, kv_cache_k, kv_cache_v, num_heads, num_kv_heads, head_dim
    )
    
    # 3. Residual
    hidden = x + attn_out
    
    # 4. Post-attention RMSNorm
    norm_hidden = rms_norm(hidden, w['post_attention_layernorm'])
    
    # 5. MLP
    gate = np.dot(w['gate_proj'], norm_hidden)
    up = np.dot(w['up_proj'], norm_hidden)
    mlp_out = np.dot(w['down_proj'], geglu(gate, up))
    
    # 6. Residual
    return hidden + mlp_out

def main():
    print("--- Reference Attention ---")
    hidden_size = 4
    num_heads = 2
    num_kv_heads = 1
    head_dim = 2
    
    x = np.ones(hidden_size, dtype=np.float32)
    pos = 0
    q_proj = np.ones((num_heads * head_dim, hidden_size), dtype=np.float32)
    k_proj = np.ones((num_kv_heads * head_dim, hidden_size), dtype=np.float32)
    v_proj = np.ones((num_kv_heads * head_dim, hidden_size), dtype=np.float32)
    o_proj = np.ones((hidden_size, num_heads * head_dim), dtype=np.float32)
    
    freqs_cos = np.array([1.0], dtype=np.float32)
    freqs_sin = np.array([0.0], dtype=np.float32)
    
    max_seq_len = 10
    kv_cache_k = np.zeros((max_seq_len, num_kv_heads * head_dim), dtype=np.float32)
    kv_cache_v = np.zeros((max_seq_len, num_kv_heads * head_dim), dtype=np.float32)
    
    out = forward_attention(x, pos, q_proj, k_proj, v_proj, o_proj, freqs_cos, freqs_sin, kv_cache_k, kv_cache_v, num_heads, num_kv_heads, head_dim)
    print("attn_out:", out)
    
    print("\n--- Reference Layer ---")
    w = {
        'input_layernorm': np.ones(hidden_size, dtype=np.float32),
        'post_attention_layernorm': np.ones(hidden_size, dtype=np.float32),
        'q_proj': q_proj,
        'k_proj': k_proj,
        'v_proj': v_proj,
        'o_proj': o_proj,
        'gate_proj': np.ones((4, hidden_size), dtype=np.float32),
        'up_proj': np.ones((4, hidden_size), dtype=np.float32),
        'down_proj': np.ones((hidden_size, 4), dtype=np.float32),
    }
    
    out_layer = forward_layer(x, pos, w, freqs_cos, freqs_sin, kv_cache_k, kv_cache_v, num_heads, num_kv_heads, head_dim, 4)
    print("layer_out:", out_layer)

if __name__ == "__main__":
    main()
