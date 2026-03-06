# Flow: pure-mojo-generation

## Specification

### Goal
Implement KV-Cache management and LLM generation in Mojo. Adapt the forward pass for incremental single-token decoding. Implement Temperature, Top-K, and Nucleus sampling natively in Mojo, and expose a Python streaming generator.

### Code Analysis Summary
- We currently have `forward_sequence` for embeddings, which computes the full prompt but pools the output.
- `src/mo/core.mojo` exports a `step_mojo` function that currently returns dummy logits (`np.zeros(256000, dtype=np.float32)`).
- We need a stateful `Engine` or `KVCache` object that persists between `step_mojo` calls.
- The `generate_stream` in `src/py/mogemma/model.py` is currently implemented to yield tokens repeatedly until `eos_token_id`. It calls `_core.step()`, applies Python-side sampling (if Temp/Top-K/Top-P logic is there, or delegates to Mojo). Right now `_sample_next_token` in `model.py` handles sampling in Python. 

### Requirements
1. **KV Cache Management**: Create a persistent KV cache struct in Mojo that can hold keys and values for a full session (up to `max_sequence_length`).
2. **Incremental Decoding**: Implement an incremental forward pass `forward_step` that takes a single token, updates the KV cache at the current position, and returns the logits.
3. **Mojo Engine State**: Create an `InferenceEngine` object in Python that initializes the Mojo KV cache and holds the pointer to it, passing it to `step` on each iteration.
4. **Sampling Integration**: Depending on performance, either keep sampling in Python (currently implemented in `model.py`) and just return logits from Mojo, or implement sampling natively in Mojo to avoid returning large logit arrays back to Python. *Decision: Return raw logits from Mojo `step` first for simplicity, as Python sampling is already implemented and tested in `model.py`. We can optimize later if needed.*
5. **Python Binding Update**: Update `init_model_mojo` to return a stateful engine handle. Update `step_mojo` to accept this handle and the current token ID, execute the forward pass, and return the logits.

### Non-goals
- Continuous batching (only single sequence generation for now).
- Speculative decoding.

## Implementation Plan

### Phase 1: Stateful KV Cache
- [x] 1.1 Create `KVCache` struct in `src/mo/model.mojo` to hold the persistent cache allocations.
- [x] 1.2 Implement an `init_kv_cache` function to allocate memory.

### Phase 2: Incremental Forward Pass
- [x] 2.1 Add `forward_step` to `src/mo/layers.mojo` that processes a single token, updates the cache at `pos`, and returns logits for the final vocabulary projection.
- [x] 2.2 Add final LM Head projection (`lm_head` weight) to `ModelWeights`.

### Phase 3: Engine State & Python Bindings
- [x] 3.1 Update `init_model_mojo` to create a `KVCache` and return its memory address alongside metadata.
- [x] 3.2 Update `step_mojo` to accept the `KVCache` pointer, current `pos`, and `token_id`. It should run `forward_step` and return the logits array.

### Phase 4: Python Integration & Verification
- [x] 4.1 Update `src/py/mogemma/model.py` to maintain `pos` and pass it to `_core.step`.
- [x] 4.2 Verify generation output against expected token sequences in tests.
- [x] 4.3 Add specific tests for the generation flow.