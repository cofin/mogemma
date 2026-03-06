# Flow: embedding-model-id-policy

## Specification
Validation currently references an embedding model ID that is not resolvable from the public `gemma-data` bucket.
This flow will remove non-existent default embedding IDs from baseline validation while keeping explicit errors when users request unavailable IDs.

### Verification Snapshot (2026-02-24)
- `checkpoints/gemma3-text-embedding-4m/` returned `COUNT 0` via `obstore` listing.
- Control checks returned non-zero entries for `checkpoints/gemma3-270m-it/` and `checkpoints/gemma3n-e2b-it/`.

### Code Analysis Summary
- `tools/validate.py` currently includes `"gemma3-text-embedding-4m"` in default embedding models list (`models_to_test_embed`), causing default validation failure.
- Model resolution is intentionally strict in `src/py/mogemma/model.py` via `HubManager.resolve_model(..., download_if_missing=True, strict=True)`.
- `src/py/mogemma/hub.py` raises `ModelNotFoundError` with actionable message when a checkpoint path is absent in public bucket.
- Existing tests already expect this fail-fast behavior for invalid IDs in:
  - `src/py/tests/test_hub.py`
  - `src/py/tests/test_embeddings.py`

### Requirements
1. Remove non-resolvable model IDs from default embedding validation set.
2. Preserve strict explicit failures for user-specified unavailable model IDs.
3. Add regression coverage for both valid default IDs and invalid explicit IDs.
4. Keep diagnostics actionable and consistent with existing hub error contracts.

## Implementation Plan

### Phase 1: Model-ID Policy
- [x] 1.1 `mogemma-1bx.1` Remove `gemma3-text-embedding-4m` from default embedding validation list. [2a33c99]
- [x] 1.2 `mogemma-1bx.2` Keep default embedding coverage on resolvable IDs only (`gemma3-270m-it`, `gemma3n-e2b-it` unless superseded by later discovery). [2a33c99]

### Phase 2: Fail-Fast Contracts
- [x] 2.1 `mogemma-1bx.3` Preserve actionable `ModelNotFoundError` behavior for explicit invalid `--model` values. [e114837]
- [x] 2.2 `mogemma-1bx.4` Add regression checks for valid defaults and invalid explicit IDs. [e114837]
