# Flow: gcs-download-backend

## Specification

### Goal
Replace the HuggingFace HTTP download backend in `hub.py` with direct GCS
downloads from the public `gemma-data` bucket using `obstore.store.GCSStore`.

### Code Analysis Summary

**Files to modify:**
- `src/py/mogemma/hub.py` (primary — rewrite download methods)

**Files to create:**
- `src/py/mogemma/orbax_loader.py` (resurrect for Gemma 4)

**Files to update tests:**
- `src/py/tests/test_hf_download.py` → rename to `test_gcs_download.py`
- `src/py/tests/test_hub.py` (update mocks)

**Key interfaces preserved:**
- `HubManager.download_sync(model_id) -> Path`
- `HubManager.download_async(model_id) -> Path`
- `HubManager.resolve_model(model_id, ...) -> Path`
- `HubManager.resolve_model_async(model_id, ...) -> Path`

**GCS conventions (from old code):**
- Store: `GCSStore("gemma-data", config={"skip_signature": "true"})`
- Checkpoint prefix: `checkpoints/{clean_id}/` (e.g., `checkpoints/gemma4-e2b-it/`)
- Tokenizer: `tokenizers/tokenizer_gemma4.model`
- File enumeration: `obs.list(store, prefix)` returns all files in checkpoint dir

**Orbax checkpoint structure (OCDBT):**
- Root contains: `_METADATA`, `_CHECKPOINT_METADATA`, `manifest.ocdbt`, `descriptor/`, `d/`
- Tensor data in: `ocdbt.process_0/` subdir with `manifest.ocdbt` and `d/` containing shards
- Detection: `(path / "ocdbt.process_0").is_dir() and (path / "manifest.ocdbt").exists()`

### Requirements

1. `_make_gcs_store()` replaces `_make_hf_store()` — returns `GCSStore` with `skip_signature`
2. Download enumerates all files via `obs.list()` then downloads each via `obs.get()`
3. Tokenizer downloaded from `tokenizers/tokenizer_gemma4.model` (bucket root, not checkpoint dir)
4. Cache structure unchanged: `~/.cache/mogemma/{model_id_with_dashes}/`
5. `_has_model_files()` detects Orbax checkpoints (`ocdbt.process_0/` + `manifest.ocdbt`) 
   in addition to safetensors (for backward compat with existing cached models)
6. OrbaxLoader resurrected with minimal interface: `can_load()`, `get_tensor_metadata()`, `close()`
7. `auto_loader()` in `loader.py` updated to try `OrbaxLoader` as fallback
8. `_clean_model_id()` mapping: `"google/gemma-4-E2B-it"` → `"gemma4-e2b-it"` (lowercase for GCS paths)

---

## Implementation Plan

### Phase 1: GCS Store + Download Rewrite
*Focus: Replace HF download methods with GCS equivalents*

- [ ] 1.1 **Replace store factory** in `hub.py`
  - Delete: `_make_hf_store()`, `_get_hf_token()`, `_hf_resolve_url()`, `_HF_BASE`
  - Add: `_make_gcs_store() -> GCSStore` returning `GCSStore("gemma-data", config={"skip_signature": "true"})`
  - Update import: `from obstore.store import GCSStore, LocalStore` (remove `HTTPStore`)
  - File: `src/py/mogemma/hub.py:14,40-59`

- [ ] 1.2 **Add GCS path helpers**
  - Add: `_gcs_checkpoint_prefix(clean_id: str) -> str` returning `f"checkpoints/{clean_id}/"`
  - Add: `_gcs_tokenizer_path() -> str` returning `"tokenizers/tokenizer_gemma4.model"`
  - Update `_clean_model_id()` to also lowercase the result for GCS path matching
  - File: `src/py/mogemma/hub.py:61-71`

- [ ] 1.3 **Rewrite file enumeration**
  - Delete: `_fetch_index_json()`, `_fetch_index_json_async()`, `_parse_shard_filenames()`
  - Add: `_list_remote_files(store, prefix) -> list[str]` using `obs.list(store, prefix)`
  - Add: `_list_remote_files_async(store, prefix) -> list[str]` using `obs.list_async()`
  - Must filter out `_$folder$` marker objects
  - File: `src/py/mogemma/hub.py:111-120,266-282`

- [ ] 1.4 **Rewrite download_sync()**
  - Replace HF-specific flow with:
    1. Create GCS store
    2. List all files under `checkpoints/{clean_id}/`
    3. Download each file preserving directory structure into staging dir
    4. Download tokenizer from `tokenizers/tokenizer_gemma4.model`
    5. Finalize (move staging → cache)
  - Keep atomic staging dir pattern (tmpdir → rename)
  - File: `src/py/mogemma/hub.py:325-366`

- [ ] 1.5 **Rewrite download_async()**
  - Async equivalent of 1.4 using `obs.get_async()` and `obs.list_async()`
  - Use `asyncio.gather()` for parallel shard downloads
  - File: `src/py/mogemma/hub.py:368-415`

### Phase 2: OrbaxLoader
*Focus: Resurrect OrbaxLoader for Gemma 4 checkpoint reading*

- [ ] 2.1 **Create `src/py/mogemma/orbax_loader.py`**
  - Based on deleted version from commit `9aac18e` but updated for Gemma 4
  - **Download layer uses obstore GCSStore** (no gcsfs dependency)
  - **Local OCDBT reading uses tensorstore** (already used by old OrbaxLoader — tensorstore
    is the only library that can read the OCDBT key-value format)
  - Dependencies: `tensorstore` (for local OCDBT reading), `numpy`
  - Key methods:
    - `__init__(model_path)` — opens local OCDBT store via tensorstore, enumerates tensors, loads into memory
    - `can_load(model_path) -> bool` — checks for `ocdbt.process_0/` + `manifest.ocdbt`
    - `get_tensor_metadata() -> dict[str, tuple[int, tuple[int, ...], str]]` — same interface as SafetensorsLoader
    - `close()` — releases numpy arrays
    - Context manager support (`__enter__`/`__exit__`)
  - TensorStore spec for **local** OCDBT (files already downloaded by obstore GCSStore):
    ```python
    {
        "driver": "zarr",
        "kvstore": {
            "driver": "ocdbt",
            "base": f"file://{model_path}/ocdbt.process_0/",
            "path": tensor_name,
        },
    }
    ```
  - Enumerate tensor names via OCDBT kvstore `.list()` filtering for `.zarray` entries
  - BF16→F32 conversion inline (same as SafetensorsLoader)
  - **Note:** obstore handles GCS downloads (Chapter 1 Phase 1), tensorstore handles
    local OCDBT format reading. These are separate concerns — no gcsfs involved.

- [ ] 2.2 **Update `auto_loader()` in `loader.py`**
  - Add `OrbaxLoader` import (lazy, inside function)
  - Try `SafetensorsLoader.can_load()` first, then `OrbaxLoader.can_load()`
  - Update error message to mention both formats
  - File: `src/py/mogemma/loader.py:207-215`

- [ ] 2.3 **Update `_has_model_files()` in `hub.py`**
  - Add Orbax detection: `(path / "ocdbt.process_0").is_dir() and (path / "manifest.ocdbt").exists()`
  - Keep safetensors check for backward compatibility
  - File: `src/py/mogemma/hub.py:106-109`

### Phase 3: Update Hub Resolve + Docstrings
*Focus: Update resolve methods and error messages*

- [ ] 3.1 **Update error messages and docstrings**
  - Module docstring: `"Model resolution and GCS download helpers."`
  - Class docstring: `"Manages downloading and caching Gemma 4 models from Google Cloud Storage."`
  - Error class: `GCSDownloadError` replaces `HFDownloadError`
  - `ModelNotFoundError` message: reference `gemma-data` bucket, not HuggingFace
  - `resolve_model()` strict error: reference GCS, not HF
  - File: `src/py/mogemma/hub.py:1,22-23,219-223,255-260`

- [ ] 3.2 **Update `_get_tokenizer_path()` and `resolve_tokenizer()`**
  - `_get_tokenizer_path()` returns `"tokenizer.model"` (local cache name)
  - GCS source path handled by download methods (always `tokenizers/tokenizer_gemma4.model`)
  - `resolve_tokenizer()` error message: reference GCS download
  - File: `src/py/mogemma/hub.py:159-193`

- [ ] 3.3 **Update `_finalize_download()`**
  - Validate presence of Orbax files OR safetensors (either format valid after download)
  - File: `src/py/mogemma/hub.py:308-323`

### Phase 4: Tests
*Focus: Rewrite tests for GCS backend*

- [ ] 4.1 **Rewrite `test_hf_download.py` → `test_gcs_download.py`**
  - `TestMakeGCSStore`: verify store creation with skip_signature
  - `TestGCSPathHelpers`: test `_gcs_checkpoint_prefix()`, `_gcs_tokenizer_path()`
  - `TestCleanModelId`: test lowercase normalization for GCS paths
  - `TestListRemoteFiles`: mock `obs.list()` to return checkpoint file list
  - `TestDeletedHFCode`: verify HF methods no longer exist
  - File: `src/py/tests/test_gcs_download.py` (new, replacing `test_hf_download.py`)

- [ ] 4.2 **Update `test_hub.py`**
  - Update any mocks that reference HF-specific methods
  - File: `src/py/tests/test_hub.py`

- [ ] 4.3 **Add OrbaxLoader unit tests**
  - `test_can_load()`: mock directory structure detection
  - `test_get_tensor_metadata()`: mock tensorstore reads, verify interface contract
  - File: `src/py/tests/test_orbax_loader.py` (new)

### Verification

- [ ] `make lint` passes (ruff check + format)
- [ ] `make test` passes (all unit tests, mocked — no network)
- [ ] Manual smoke test: `uv run python -c "from mogemma.hub import HubManager; h = HubManager(); print(h._make_gcs_store())"` succeeds
- [ ] `OrbaxLoader.can_load()` correctly detects Orbax directories vs safetensors
- [ ] `auto_loader()` routes to correct loader based on format
