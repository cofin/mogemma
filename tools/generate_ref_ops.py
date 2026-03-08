"""Generates reference outputs for the Mogemma neural network operations.

These are used to ensure correctness of the Mojo implementation.
"""

import math
import sys

import numpy as np
import numpy.typing as npt


def geglu(gate: npt.NDArray[np.float32], up: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Compute GeGLU activation.

    Args:
        gate: The gate tensor.
        up: The up projection tensor.

    Returns:
        The activated output tensor.
    """
    # Apply erf element-wise
    erf_vec = np.vectorize(math.erf)
    return 0.5 * gate * (1.0 + erf_vec(gate / np.sqrt(2.0))) * up


def rms_norm(x: npt.NDArray[np.float32], weight: npt.NDArray[np.float32], eps: float = 1e-6) -> npt.NDArray[np.float32]:
    """Compute Root Mean Square Layer Normalization.

    Args:
        x: Input tensor.
        weight: Learnable weights.
        eps: Epsilon for numerical stability.

    Returns:
        Normalized tensor.
    """
    return x / np.sqrt(np.mean(x**2) + eps) * weight


def rope_rotate(
    x: npt.NDArray[np.float32], cos: npt.NDArray[np.float32], sin: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Apply Rotary Positional Embedding.

    Args:
        x: Input tensor.
        cos: Cosine part of RoPE.
        sin: Sine part of RoPE.

    Returns:
        Rotated tensor.
    """
    # x is (D,), cos, sin are (D//2,)
    d2 = len(x) // 2
    x_part1 = x[:d2]
    x_part2 = x[d2:]
    out = np.zeros_like(x)
    out[:d2] = x_part1 * cos - x_part2 * sin
    out[d2:] = x_part1 * sin + x_part2 * cos
    return out


def softmax(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Compute stable softmax.

    Args:
        x: Input tensor.

    Returns:
        Softmax normalized tensor.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def forward_attention(  # noqa: PLR0913
    x: npt.NDArray[np.float32],
    pos: int,
    q_proj: npt.NDArray[np.float32],
    k_proj: npt.NDArray[np.float32],
    v_proj: npt.NDArray[np.float32],
    o_proj: npt.NDArray[np.float32],
    freqs_cos: npt.NDArray[np.float32],
    freqs_sin: npt.NDArray[np.float32],
    kv_cache_k: npt.NDArray[np.float32],
    kv_cache_v: npt.NDArray[np.float32],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> npt.NDArray[np.float32]:
    """Simulate a single self-attention step.

    Args:
        x: Input hidden states.
        pos: Sequence position.
        q_proj: Q weights.
        k_proj: K weights.
        v_proj: V weights.
        o_proj: Output weights.
        freqs_cos: RoPE cosine.
        freqs_sin: RoPE sine.
        kv_cache_k: Key cache.
        kv_cache_v: Value cache.
        num_heads: Number of query heads.
        num_kv_heads: Number of kv heads.
        head_dim: Head dimension.

    Returns:
        Attention output tensor.
    """
    # Q, K, V Projections
    q = np.dot(q_proj, x)  # (num_heads * head_dim,)
    k = np.dot(k_proj, x)  # (num_kv_heads * head_dim,)
    v = np.dot(v_proj, x)  # (num_kv_heads * head_dim,)

    # Reshape for heads
    q = q.reshape(num_heads, head_dim)
    k = k.reshape(num_kv_heads, head_dim)
    v = v.reshape(num_kv_heads, head_dim)

    # RoPE
    for h in range(num_heads):
        q[h] = rope_rotate(q[h], freqs_cos, freqs_sin)
    for h in range(num_kv_heads):
        k[h] = rope_rotate(k[h], freqs_cos, freqs_sin)

    # Update cache
    kv_cache_k[pos] = k
    kv_cache_v[pos] = v

    # Attention
    attn_out = np.zeros_like(q)
    group_size = num_heads // num_kv_heads

    for h in range(num_heads):
        kv_h = h // group_size
        q_h = q[h]
        # Query against all cached K up to pos
        keys = kv_cache_k[: pos + 1, kv_h, :]  # (pos+1, head_dim)
        scores = np.dot(keys, q_h) / np.sqrt(head_dim)  # (pos+1,)
        probs = softmax(scores)

        # Weighted sum of cached V
        vals = kv_cache_v[: pos + 1, kv_h, :]  # (pos+1, head_dim)
        attn_out[h] = np.dot(probs, vals)

    # Output projection
    attn_out = attn_out.reshape(-1)
    return np.dot(o_proj, attn_out)


def forward_layer(  # noqa: PLR0913
    x: npt.NDArray[np.float32],
    pos: int,
    w: dict[str, npt.NDArray[np.float32]],
    freqs_cos: npt.NDArray[np.float32],
    freqs_sin: npt.NDArray[np.float32],
    kv_cache_k: npt.NDArray[np.float32],
    kv_cache_v: npt.NDArray[np.float32],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> npt.NDArray[np.float32]:
    """Simulate a single transformer layer step.

    Args:
        x: Input hidden states.
        pos: Sequence position.
        w: Dictionary of weights.
        freqs_cos: RoPE cosine.
        freqs_sin: RoPE sine.
        kv_cache_k: Key cache.
        kv_cache_v: Value cache.
        num_heads: Number of query heads.
        num_kv_heads: Number of kv heads.
        head_dim: Head dimension.

    Returns:
        Layer output tensor.
    """
    # 1. Input RMSNorm
    norm_x = rms_norm(x, w["input_layernorm"])

    # 2. Attention
    attn_out = forward_attention(
        norm_x,
        pos,
        w["q_proj"],
        w["k_proj"],
        w["v_proj"],
        w["o_proj"],
        freqs_cos,
        freqs_sin,
        kv_cache_k,
        kv_cache_v,
        num_heads,
        num_kv_heads,
        head_dim,
    )

    # 3. Residual
    hidden = x + attn_out

    # 4. Post-Attention RMSNorm
    norm_hidden = rms_norm(hidden, w["post_attention_layernorm"])

    # 5. MLP
    gate = np.dot(w["gate_proj"], norm_hidden)
    up = np.dot(w["up_proj"], norm_hidden)
    mlp_activation = geglu(gate, up)
    mlp_out = np.dot(w["down_proj"], mlp_activation)

    # 6. Residual
    return hidden + mlp_out


def main() -> None:
    """Generate reference operations for validation."""
    sys.stdout.write("--- Reference Attention ---\n")
    hidden_size = 4
    num_heads = 2
    num_kv_heads = 1
    head_dim = 2

    # Dummy data
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    pos = 0

    # Weights (using simple patterns for deterministic outputs)
    q_proj = np.ones((num_heads * head_dim, hidden_size), dtype=np.float32)
    k_proj = np.ones((num_kv_heads * head_dim, hidden_size), dtype=np.float32)
    v_proj = np.ones((num_kv_heads * head_dim, hidden_size), dtype=np.float32)
    o_proj = np.ones((hidden_size, num_heads * head_dim), dtype=np.float32)

    freqs_cos = np.array([1.0], dtype=np.float32)
    freqs_sin = np.array([0.0], dtype=np.float32)

    # Initialize cache storage
    kv_cache_k = np.zeros((10, num_kv_heads, head_dim), dtype=np.float32)
    kv_cache_v = np.zeros((10, num_kv_heads, head_dim), dtype=np.float32)

    out = forward_attention(
        x,
        pos,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        freqs_cos,
        freqs_sin,
        kv_cache_k,
        kv_cache_v,
        num_heads,
        num_kv_heads,
        head_dim,
    )
    sys.stdout.write(f"attn_out: {out}\n")

    sys.stdout.write("\n--- Reference Layer ---\n")
    w = {
        "input_layernorm": np.ones(hidden_size, dtype=np.float32),
        "q_proj": q_proj,
        "k_proj": k_proj,
        "v_proj": v_proj,
        "o_proj": o_proj,
        "post_attention_layernorm": np.ones(hidden_size, dtype=np.float32),
        "gate_proj": np.ones((8, hidden_size), dtype=np.float32),
        "up_proj": np.ones((8, hidden_size), dtype=np.float32),
        "down_proj": np.ones((hidden_size, 8), dtype=np.float32),
    }

    out_layer = forward_layer(
        x, pos, w, freqs_cos, freqs_sin, kv_cache_k, kv_cache_v, num_heads, num_kv_heads, head_dim
    )
    sys.stdout.write(f"layer_out: {out_layer}\n")


if __name__ == "__main__":
    main()
