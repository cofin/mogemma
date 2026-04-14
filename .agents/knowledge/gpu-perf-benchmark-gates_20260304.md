# Learnings for Flow: gpu-perf-benchmark-gates_20260304

## Summary
Completed the following key tasks:
  - 1.1 Define benchmark CLI/options contract, including the canonical token-count flag, any temporary alias/deprecation behavior, backend/device flags, and explicit synthetic-vs-real mode selection.
  - 1.2 Define JSON schema v2 for benchmark output, enumerating required top-level fields, environment fields, threshold metadata, and per-mode metrics for both generation and embedding.
  - 1.3 Define strict separation between synthetic (stub) and real-backend benchmark modes, including which code paths may install stubs, which modes require cached real assets, and how outputs label that distinction.
  - 1.4 Checkpoint: approve schema and CLI contract before implementation.
  - 2.1 Define baseline loading/comparison algorithm keyed by `(mode, backend, runner_class, model_tier)` for both generation and embedding metrics.
  - ... and 11 more tasks.

## Patterns & Gotchas
- **Architecture**: Ensure components are fully aligned with project conventions outlined in `.agent/patterns.md`.
- **Validation**: Rely on empirical testing and standard parity gates before finalizing flows.
