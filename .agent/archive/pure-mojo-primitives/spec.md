# Flow: pure-mojo-primitives

## Specification

### Goal
Implement native Mojo math primitives required for the Gemma 3 transformer forward pass, specifically:
- `RMSNorm` (Root Mean Square Normalization)
- `GeGLU` (GELU activation with a linear projection gate)
- `RoPE` (Rotary Positional Embeddings)
- `matmul` (Matrix multiplication using `linalg` or SIMD for linear projections)

### Code Analysis Summary
- In Chapter 1 (`pure-mojo-bridge`), we successfully eliminated the `modular`/`max` dependencies.
- `src/mo/core.mojo` now has a pointer-based `init_model_mojo` which creates zero-copy `Tensor` views of `safetensors`.
- The current `step_mojo` and `generate_embeddings_mojo` stubs return zero-filled or random data. We must build the math functions that will replace these stubs with real forward pass operations in Chapter 3.
- `src/mo/` needs a new module or file(s) for the math layers (e.g., `src/mo/math.mojo` or `src/mo/primitives.mojo`).

### Requirements
1. Implement `RMSNorm` following the standard LLaMA/Gemma definition (scale by variance, multiply by weight, plus epsilon).
2. Implement `GeGLU` following the Gemma 3 specification (using GELU approximation or exact calculation, multiplying by the gated input).
3. Implement `RoPE` to rotate queries and keys based on their positions.
4. Set up an optimized `matmul` operation that can multiply query/key/value tensors, applying appropriate scaling.
5. All operations should utilize Mojo's `SIMD` and `math` libraries for maximum performance.
6. Create unit tests for these primitives to ensure accuracy against known numpy/python outputs.

### Non-goals
- Assembling the full transformer block or computing self-attention (this belongs to Chapter 3: `pure-mojo-transformer`).
- KV cache manipulation (this belongs to Chapter 4: `pure-mojo-generation`).

## Implementation Plan

### Phase 1: Tensor Utilities and Math Setup
- [x] 1.1 Create `src/mo/math.mojo` or similar.
- [x] 1.2 Implement utility functions for `Tensor` manipulation if needed.

### Phase 2: Activation and Normalization (RMSNorm, GeGLU)
- [x] 2.1 Implement `RMSNorm` with SIMD-vectorized variance computation.
- [x] 2.2 Implement `GeGLU` using `math.erf` or an approximation, applied via SIMD.

### Phase 3: Positional Embeddings (RoPE)
- [x] 3.1 Implement `RoPE` for query and key rotations.
- [x] 3.2 Ensure frequency precomputation is properly handled or passed.

### Phase 4: Matrix Multiplication
- [x] 4.1 Implement `matmul` (or leverage `algorithm.vectorize` / `linalg.matmul` if available in the standard library) to handle standard linear projections.

### Phase 5: Verification
- [x] 5.1 Add Mojo tests to verify numerical accuracy of `RMSNorm`, `GeGLU`, `RoPE`, and `matmul`.
- [x] 5.2 Add a Python script to generate reference values for the Mojo tests to validate against.
