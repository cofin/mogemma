# Knowledge: Pure Mojo Generation (KV-Cache & Sampling)

> Flow: pure-mojo-generation | Archived 2026-02-25

## Key Changes

Created `KVCache` struct for persistent cache allocations. Implemented incremental single-token decoding via `forward_step`. Sampling kept in Python; logits returned from Mojo.

## Patterns

- **Stateful engine:** `init_model_mojo` initializes cache; `step_mojo` iterates with persistent state across FFI boundary.
- **Incremental forward pass:** Updates cache at current position, returns raw logits.
- **Sampling in Python:** Temperature, Top-K, Nucleus handled in Python (revisit if bottleneck).

## Gotchas

- KV cache must persist between `step_mojo` calls — stateful FFI boundary.
- Cache position management is critical for correctness.
- Single-token-at-a-time only (no batching). Continuous batching and speculative decoding deferred.
