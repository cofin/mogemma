# Flow: core-embedding-padding

## Specification
The `tools/validate.py` script fails during embedding generation when passed inputs of varying lengths. The `EmbeddingModel.embed` Python wrapper attempts to pack token IDs of different lengths into a 2D NumPy array. Because Python padding was disabled (`_Tokenizer.enable_padding()` is a no-op), this causes `numpy.asarray` to fail with "inhomogeneous shape".

Instead of padding in Python, we will push the padding logic down into the Mojo `_core` implementation. This is the most performant approach as it allows the backend to handle jagged arrays internally and construct optimized padded batches in C/Mojo.

### Code Analysis Summary
- `src/py/mogemma/model.py`: The `embed` method currently calls `self._embed_token_array(tokens, len(text))` passing a NumPy array of `int32`.
- `tools/validate.py`: Passes `["The quick brown fox jumps over the lazy dog.", "MAX Engine is fast."]` which have different token lengths.
- The `generate_embeddings` function in Mojo (`_core.so`) needs to accept a list of 1D arrays or pad the incoming numpy arrays appropriately.

## Implementation Plan

### Phase 1: Update Python Wrapper to Handle Jagged Token Arrays
- [ ] 1.1 In `src/py/mogemma/model.py`, modify `EmbeddingModel.embed` to prevent forcing jagged lists into a 2D `np.asarray` if they have different lengths.
- [ ] 1.2 Update the `_embed_token_array` method or the FFI boundary to pass a list of 1D numpy arrays (or flat array with offsets) down to `_core.generate_embeddings`.

### Phase 2: Implement Padding in Mojo Core
- [ ] 2.1 Update `src/mo/mogemma/core.mojo` (or `ops.mojo`) where `generate_embeddings` is implemented to accept the new jagged array format (e.g., list of vectors or flattened vector + lengths).
- [ ] 2.2 Implement internal padding/masking logic in Mojo to pad batches to the maximum sequence length dynamically before executing the forward pass.
- [ ] 2.3 Ensure the returned embedding matrix is `[batch_size, embed_dim]`.

### Phase 3: Validation
- [ ] 3.1 Run `uv run tools/validate.py --mode embed` to verify the "inhomogeneous shape" crash is fixed.
- [ ] 3.2 Ensure the returned embeddings are valid and the shape matches `(2, embed_dim)`.