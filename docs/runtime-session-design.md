# Runtime Session Design Notes

## Current State Transitions (Baseline)

1. `init_model(metadata)`:
   - Detects architecture (`standard` vs `nano`).
   - Builds runtime metadata object and computes static dimensions.
   - Allocates long-lived numpy caches (`k_cache`, `v_cache`, `freqs_cos`, `freqs_sin`).
   - Stores mutable position counter (`pos`) and runtime metadata in `llm` dictionary.
2. `step(llm, token, sampling...)`:
   - Reads mutable position/caches from `llm`.
   - Rebuilds model-view (`_build_model_from_runtime` / `_build_nano_model_from_runtime`) on each token.
   - Allocates transient scratch and output logits each call.
   - Advances `llm[\"pos\"]`.
3. `generate_embeddings(llm, batch_tokens)`:
   - Reads runtime metadata.
   - Rebuilds model-view per call.
   - Re-allocates RoPE, KV, scratch, and intermediate embedding buffers per call.

## Allocation Boundary Snapshot

### Long-lived (session-scoped)

- `runtime` metadata object
- `k_cache`, `v_cache`
- precomputed RoPE arrays in `init_model` (`freqs_cos`, `freqs_sin`)
- `pos`, dimension metadata (`num_layers`, `hidden_size`, ...)

### Transient (call-scoped)

- `step`: scratch buffer, logits buffer
- `generate_embeddings`: RoPE lists, KV lists, scratch, input_ids, emb_out
- model-view materialization (`ModelWeights` / `NanoModelWeights`) in both `step` and `generate_embeddings`

## Proposed Runtime/Session Structure

### Runtime Descriptor (immutable after init)

- Owner: `init_model` stage
- Contents:
  - architecture kind (`standard` / `nano`)
  - tensor metadata + static dimensions
  - optional backend capabilities flags (future)
- Goal: single source of shape/layout truth across all entrypoints

### Session State (mutable, reusable)

- Owner: `step` + session lifecycle operations
- Contents:
  - position (`pos`)
  - KV caches
  - reusable RoPE/precompute buffers
  - counters/diagnostics for build and reuse checks
- Goal: avoid per-token model-view reconstruction and isolate request/session state

### Embedding Invocation State (ephemeral)

- Owner: `generate_embeddings`
- Contents:
  - per-call scratch/input/output working buffers
- Goal: keep embedding path deterministic without leaking mutable generation state

## Mutable Field Ownership Map (Checkpoint)

- `pos`: session state owner (resettable by lifecycle hook)
- `k_cache` / `v_cache`: session state owner (resettable/zeroable)
- `runtime` metadata: runtime descriptor owner (read-only post init)
- shape/config metadata (`num_layers`, `head_dim`, `hidden_size`, `intermediate_size`, `vocab_size`, `per_layer_dim`, `kv_share_start`):
  - runtime descriptor owner (read-only post init)
- temporary scratch + output tensors:
  - entrypoint-local owner (`step` or `generate_embeddings`)

## Manual Verification (Controlled Multi-Prompt Sequence, 2026-03-05)

One loaded model instance was exercised with two consecutive prompts and one embedding batch in the same process (local stubbed core path used for deterministic inspection).

Observed checkpoints:

- `responses_equal = True` for repeated deterministic decode path
- `first_positions = [0, 0]` (each generation starts at session position 0)
- `k_cache_zero = True` after generation reset hook
- `embedding_shape = (2, 8)` on same loaded session context

This confirms session reuse hooks execute without cross-prompt state drift for the current CPU abstraction path.
