# Knowledge: Pure Mojo Math Primitives

> Flow: pure-mojo-primitives | Archived 2026-02-25

## Key Changes

Implemented four core Mojo math primitives: RMSNorm, GeGLU, RoPE, matmul. All use SIMD vectorization.

## Patterns

- **SIMD-vectorized variance** for RMSNorm computation.
- **RoPE frequency precomputation** for efficiency (precompute vs on-the-fly tradeoff).
- **Reference validation:** Every Mojo implementation tested against Python/NumPy baselines.

## Gotchas

- GELU approximation vs exact affects numerical accuracy — must match Gemma 3 spec exactly.
- RoPE frequency management needs careful handling across sequence positions.
