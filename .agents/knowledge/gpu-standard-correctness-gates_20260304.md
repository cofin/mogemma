# Learnings for Flow: gpu-standard-correctness-gates_20260304

## Summary
Completed the following key tasks:
  - 1.1 Define the exact correctness dimensions to gate: primitive outputs, layer outputs, `_core.step` logits/counters, and end-to-end deterministic generation tokens.
  - 1.2 Freeze deterministic execution policy, including prompt corpus, instruction formatting rules, seeds if any, and backend/fallback reporting expectations.
  - 1.3 Define baseline collection requirements: git SHA, Mojo version, Python version, CPU/GPU model, driver/runtime version, prompt corpus ID, prompt lengths, and token counts.
  - 1.4 Checkpoint: sign off gate contract, thresholds, and evidence schema before test implementation begins.
  - 2.1 Expand `src/mo/tests/test_ops.mojo` with backend-aware primitive parity cases and explicit tolerance assertions per primitive.
  - ... and 19 more tasks.

## Patterns & Gotchas
- **Architecture**: Ensure components are fully aligned with project conventions outlined in `.agent/patterns.md`.
- **Validation**: Rely on empirical testing and standard parity gates before finalizing flows.
