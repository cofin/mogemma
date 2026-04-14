# Learnings for Flow: gpu-nano-attention-altup_20260304

## Summary
Completed the following key tasks:
  - 1.1 Define explicit kernel contracts for nano attention, AltUp predict, AltUp correct, Laurel, and per-layer mapping (inputs, outputs, strides, dtypes, scratch ownership, precomputed tables, parity target).
  - 1.2 Define the GPU dispatch seam in `core.mojo`/`layers.mojo` without changing public Python API, and make the backend choice session-latched rather than per-call.
  - 1.3 Checkpoint: produce an ownership/contract table mapping each CPU function to its GPU equivalent, touched module, and frozen parity tolerance.
  - 2.1 Add failing deterministic parity tests for `forward_attention_nano` (including `write_kv=True/False`).
  - 2.2 Add failing deterministic parity tests for AltUp predict and AltUp correct routing (shape + value checks, modality ordering, coefficient-axis expectations).
  - ... and 16 more tasks.

## Patterns & Gotchas
- **Architecture**: Ensure components are fully aligned with project conventions outlined in `.agent/patterns.md`.
- **Validation**: Rely on empirical testing and standard parity gates before finalizing flows.
