# Flow: tokenizer-migration

## Specification

### Code Analysis Summary
I reviewed the current state of the codebase to understand the dependency on `transformers` and what needs to be changed for "Tokenizer Liberation".
- **Files Analyzed**: `src/py/mogemma/model.py`, `pyproject.toml`, and tests (`test_gemma_model.py`, `test_embeddings.py`, `test_async.py`, `test_vision_model.py`).
- **Dependencies**: `transformers>=4.38.0` is present in the `text`, `vision`, and `all` optional dependency groups in `pyproject.toml`.
- **Usage in `model.py`**: `AutoTokenizer` is imported conditionally and used in `_ensure_tokenizer` methods of `EmbeddingModel` and `GemmaModel`. The `embed` and `generate_stream` methods call the tokenizer with arguments `padding=True, truncation=True, max_length=self.config.max_sequence_length, return_tensors="np"` which are specific to the `transformers` API. The token ids are accessed via `encoded["input_ids"]`.
- **Decoding**: `generate_stream` uses `tokenizer.decode([next_token], skip_special_tokens=True)`.
- **Tests**: Mocking of `AutoTokenizer.from_pretrained` is heavily used across the test suite and tests simulate the `{"input_ids": ...}` return format.

### Requirements
- Replace `transformers` with the standalone `tokenizers` library (`tokenizers>=0.15.0`).
- Ensure no dependency on `transformers` remains.
- Update `pyproject.toml` to reflect this change.
- Refactor `model.py` to use `tokenizers.Tokenizer.from_pretrained` instead of `AutoTokenizer`.
- Update encoding calls to use `tokenizer.encode_batch` or `tokenizer.encode`. The `tokenizers` library requires explicit padding/truncation configuration and returns `Encoding` objects where token IDs are accessed via the `.ids` attribute.
- Update decoding calls to use `tokenizer.decode`.
- Update all tests to mock the new tokenizer and its return types properly.

## Implementation Plan

### Phase 1: Update Dependencies
- [ ] 1.1 In `pyproject.toml`, remove `"transformers>=4.38.0"` from `text`, `vision`, and `all` groups.
- [ ] 1.2 In `pyproject.toml`, add `"tokenizers>=0.15.0"` to `text`, `vision`, and `all` groups.
- [ ] 1.3 Run `uv sync` to update the lockfile with the new dependencies.

### Phase 2: Refactor `model.py` Tokenizer Import and Initialization
- [ ] 2.1 In `src/py/mogemma/model.py`, replace `from transformers import AutoTokenizer` with `from tokenizers import Tokenizer` (handle `ModuleNotFoundError` by setting `Tokenizer = None`).
- [ ] 2.2 Update `_ensure_tokenizer` in both `EmbeddingModel` and `GemmaModel` to raise `ModuleNotFoundError` mentioning `tokenizers` instead of `transformers`.
- [ ] 2.3 Update `_ensure_tokenizer` to load using `Tokenizer.from_pretrained(str(self.model_path))`.

### Phase 3: Refactor Encoding and Decoding Logic
- [ ] 3.1 Update `EmbeddingModel.embed`:
  - Configure tokenizer padding and truncation using `tokenizer.enable_truncation(max_length=self.config.max_sequence_length)` and `tokenizer.enable_padding()`.
  - Use `tokenizer.encode_batch(text)` to get encoding objects.
  - Extract IDs: `[e.ids for e in encoded]` and convert to `np.asarray(..., dtype=np.int32)`.
- [ ] 3.2 Update `GemmaModel.generate_stream`:
  - Configure tokenizer padding and truncation similarly.
  - Use `tokenizer.encode(prompt)` and get `.ids`.
  - Handle `eos_token_id` retrieval (the `tokenizers` library might need `tokenizer.token_to_id("<eos>")`).
  - Update decoding from `tokenizer.decode([next_token], skip_special_tokens=True)` to the corresponding `tokenizers` method (e.g., `tokenizer.decode([next_token])`).

### Phase 4: Fix Test Suite
- [ ] 4.1 In `src/py/tests/test_gemma_model.py`, change `patch("mogemma.model.AutoTokenizer.from_pretrained")` to patch `mogemma.model.Tokenizer.from_pretrained`.
- [ ] 4.2 Update the mock return value to match `tokenizers` behavior (an object with an `.ids` property instead of a dictionary with `input_ids`).
- [ ] 4.3 Repeat steps 4.1 and 4.2 for `test_embeddings.py`, `test_async.py`, and `test_vision_model.py`.
- [ ] 4.4 Run `uv run pytest` to ensure all tests pass.
- [ ] 4.5 Run `uv run ruff check` and `uv run pyright` to verify types and linting.
