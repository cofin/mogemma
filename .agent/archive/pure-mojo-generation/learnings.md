# Learnings: Pure Mojo Generation

## [2026-02-25] - KV-Cache & incremental decoding

- **Implemented:** `KVCache` struct, incremental `forward_step`, logits returned to Python for sampling.
- **Learnings:**
  - KV cache must persist between `step_mojo` calls — stateful FFI boundary.
  - Cache position management is critical for correctness.
  - Sampling kept in Python initially — revisit if performance bottleneck emerges.
  - Single-token-at-a-time only; batching and speculative decoding deferred.
