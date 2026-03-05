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
