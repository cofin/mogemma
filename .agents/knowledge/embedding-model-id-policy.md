# Learnings for embedding-model-id-policy

## [2026-02-25] - Model-ID Policy and Fail-Fast Contracts
- **Implemented:** Removed non-resolvable default embedding IDs (like `gemma3-text-embedding-4m`) and preserved strict failure for invalid IDs. Added explicit tests to verify `HubManager.ModelNotFoundError` is thrown with actionable message.
- **Files changed:** `tools/validate.py`, `src/py/tests/test_hub.py`, `src/py/tests/test_embeddings.py`, `src/py/tests/test_gemma_model.py`
- **Gotchas:** GCS requests inside tests (like `download_if_missing=True`) will naturally throw the proper actionable exception without needing mock overrides when the model doesn't exist.
- **Patterns:** Replaced broad `FileNotFoundError` expectations in pytest `raises` with specific `HubManager.ModelNotFoundError` to guarantee accurate contract checking.
