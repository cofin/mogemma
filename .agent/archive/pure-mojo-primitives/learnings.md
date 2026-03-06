# Learnings: Pure Mojo Math Primitives

## [2026-02-25] - Native math implementation

- **Implemented:** RMSNorm, GeGLU, RoPE, matmul — all SIMD-vectorized.
- **Learnings:**
  - GELU approximation vs exact affects numerical accuracy — must match Gemma 3 spec.
  - RoPE frequency management needs careful handling (precompute for efficiency).
  - Every Mojo implementation must be validated against Python/NumPy reference baselines.
