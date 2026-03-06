# Flow: api-max-tokens_20260225

## Specification

### Code Analysis Summary
We are replacing the parameter `max_new_tokens` in `GenerationConfig` with `max_tokens` because it is an industry-standard name (used by OpenAI, Anthropic, etc.) and avoids confusion. This change breaks backwards compatibility and touches several parts of the project, including configuration definitions, the model implementation, tests, and documentation.

- **Affected Files:**
  - `src/py/mogemma/config.py` (The main config definition and its validation logic)
  - `src/py/mogemma/model.py` (The loop boundary inside the text generation methods)
  - `src/py/tests/test_contracts_integration.py` (Instantiating `GenerationConfig`)
  - `src/py/tests/test_gemma_model.py` (Multiple instantiations of `GenerationConfig`)
  - `src/py/tests/test_async.py` (Multiple instantiations)
  - `tools/smoke_test.py`
  - `tools/validate.py`
  - `tools/benchmark.py`
  - `docs/baseline-generation.json`
  - `docs/baseline-embedding.json`
  - `README.md`

### Requirements
- Rename `max_new_tokens` to `max_tokens` in `GenerationConfig` (`src/py/mogemma/config.py`).
- Update the error message constants and validation in `__post_init__` within `config.py`.
- Update the generation loop inside `src/py/mogemma/model.py` to use `self.config.max_tokens`.
- Replace all occurrences of `max_new_tokens` with `max_tokens` across the test suite and CLI tools.
- Update all documentation files (README, baseline JSON files).
- No backwards compatibility will be provided.

## Implementation Plan

### Phase 1: API Renaming
- [x] 1.1 In `src/py/mogemma/config.py`, rename the `max_new_tokens` attribute to `max_tokens` on `GenerationConfig`. Update the corresponding `_INVALID_TOKENS_MSG` and the check inside `__post_init__`.
- [x] 1.2 In `src/py/mogemma/model.py`, update `self.config.max_new_tokens` to `self.config.max_tokens` in the generation loop.

### Phase 2: Tests and Tools Updates
- [x] 2.1 Update all tests in `src/py/tests/` to use `max_tokens` when constructing `GenerationConfig`.
- [x] 2.2 Update `tools/smoke_test.py`, `tools/validate.py`, and `tools/benchmark.py` (including CLI args) to use `max_tokens`.

### Phase 3: Documentation Updates
- [x] 3.1 Replace `max_new_tokens` with `max_tokens` in `README.md`.
- [x] 3.2 Update `docs/baseline-generation.json` and `docs/baseline-embedding.json`.