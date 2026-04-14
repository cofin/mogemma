# Learnings for Flow: gpu-kv-cache-device-memory_20260304

## Summary
Completed the following key tasks:
  - 1.1 Define the standard session-memory model: long-lived device KV buffers, long-lived step scratch, optional embedding scratch pools, and any host-visible staging buffers.
  - 1.2 Define which fields remain visible in `llm` for reset/debug (`pos`, `session_kv_cache_len`, `step_scratch_len`, `embedding_scratch_len`, active backend/debug counters) versus which stay opaque backend handles.
  - 1.3 Define the exact transfer/sync matrix for generation: token input upload, cache residency, logits egress, fallback/error path, and reset/teardown.
  - 1.4 Checkpoint: approve a lifecycle diagram covering init, steady-state step, reset, teardown, and error handling.
  - 2.1 Replace standard-session host `np.zeros` KV allocation with backend-owned device allocation while keeping CPU sessions unchanged.
  - ... and 18 more tasks.

## Patterns & Gotchas
- **Architecture**: Ensure components are fully aligned with project conventions outlined in `.agent/patterns.md`.
- **Validation**: Rely on empirical testing and standard parity gates before finalizing flows.
