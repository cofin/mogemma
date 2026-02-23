import numpy as np
from scipy.special import erf


def geglu(gate, up):
    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    gelu = 0.5 * gate * (1.0 + erf(gate / np.sqrt(2.0)))
    return gelu * up


def rms_norm(x, weight, eps=1e-6):
    mean_sq = np.mean(x**2)
    inv_rms = 1.0 / np.sqrt(mean_sq + eps)
    return x * inv_rms * weight


def rope_rotate(x, cos, sin):
    # x is shape (N,), cos and sin are shape (N//2,)
    half = len(x) // 2
    x1 = x[:half]
    x2 = x[half:]
    out = np.zeros_like(x)
    out[:half] = x1 * cos - x2 * sin
    out[half:] = x2 * cos + x1 * sin
    return out


def vec_mat_mul(x, w):
    # x is (in_dim,)
    # w is transposed (out_dim, in_dim)
    return np.dot(w, x)


def main():
    print("--- RMSNorm ---")
    x = np.ones(4, dtype=np.float32)
    w = np.ones(4, dtype=np.float32) * 2.0
    print("x:", x, "w:", w)
    print("out:", rms_norm(x, w))

    print("\n--- GeGLU ---")
    gate = np.ones(4, dtype=np.float32)
    up = np.ones(4, dtype=np.float32) * 2.0
    print("gate:", gate, "up:", up)
    print("out:", geglu(gate, up))

    print("\n--- RoPE ---")
    vec = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    cos = np.array([0.0, 1.0], dtype=np.float32)
    sin = np.array([1.0, 0.0], dtype=np.float32)
    print("vec:", vec, "cos:", cos, "sin:", sin)
    print("out:", rope_rotate(vec, cos, sin))

    print("\n--- VecMatMul ---")
    x = np.ones(4, dtype=np.float32)
    w = np.ones((2, 4), dtype=np.float32) * 2.0
    print("x:", x, "w:", w)
    print("out:", vec_mat_mul(x, w))


if __name__ == "__main__":
    main()
