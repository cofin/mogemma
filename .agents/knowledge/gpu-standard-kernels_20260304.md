# Learnings for Flow: gpu-standard-kernels_20260304

## Summary
Completed the following key tasks:
  - 1.1 Map the exact standard call graph `step_mojo -> _forward_step_standard_runtime -> forward_step -> forward_layer -> forward_attention/forward_mlp -> vec_mat_mul|rms_norm|rope_rotate|softmax|geglu`, with explicit non-goals for nano and embedding-only paths.
  - 1.2 Define ownership boundaries: `core.mojo` owns session backend latch and fallback policy, `layers.mojo` owns standard graph orchestration, and `ops.mojo` or a dedicated GPU sibling module owns backend-specific kernels.
  - 1.3 Define kernel-entry invariants: contiguous float32 pointers, scratch-layout expectations, cache pointer semantics, logits shape, and explicit prohibition on nano-only fields.
  - 1.4 Checkpoint: freeze a dispatch/invariants table that names every standard symbol that remains CPU-only versus GPU-dispatched in this chapter.
  - 2.1 Define grouped/fused GPU kernel coverage for Q/K/V projection, optional Q/K norm, RoPE, score softmax, value accumulation, output projection, RMSNorm, and GEGLU.
  - ... and 18 more tasks.

## Patterns & Gotchas
- **Architecture**: Ensure components are fully aligned with project conventions outlined in `.agent/patterns.md`.
- **Validation**: Rely on empirical testing and standard parity gates before finalizing flows.
