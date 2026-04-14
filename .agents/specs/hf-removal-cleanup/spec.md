# Flow: hf-removal-cleanup

## Specification

### Goal
Remove all HuggingFace-specific code paths, imports, tests, and references from the
codebase. After Chapters 1-3, the GCS backend is fully functional and HF code is dead.

### Depends On
- Chapters 1-3 must be complete and passing tests

### Code Analysis Summary

**HF-specific code to delete in `hub.py`** (after Chapter 1 replaces these):
- `_HF_BASE` constant (line 19)
- `_hf_resolve_url()` (lines 41-43)
- `_make_hf_store()` (lines 48-54)
- `_get_hf_token()` (lines 57-59)
- `_fetch_index_json()` (lines 266-273)
- `_fetch_index_json_async()` (lines 275-282)
- `_download_hf_file()` (lines 284-294)
- `_download_hf_file_async()` (lines 296-306)
- `_parse_shard_filenames()` (lines 113-120)
- `HTTPStore` import (line 15)
- `HFDownloadError` class (lines 219-220)

**HF-specific test files to delete:**
- `src/py/tests/test_hf_download.py` (entire file — 170 lines)
- `src/py/tests/test_hf_token.py` (entire file — ~70 lines)

**HF references in other files:**
- `hub.py:1` — module docstring mentions "HuggingFace"
- `hub.py:23` — class docstring mentions "HuggingFace"
- `hub.py:81` — `_head_exists` signature references `HTTPStore`
- `hub.py:188-192` — tokenizer error message mentions "HuggingFace download"
- `hub.py:258,453` — error messages mention "HuggingFace model id"
- `model.py:110` — docstring mentions "HuggingFace config.json"
- `src/py/tests/test_tokenizer_resolution.py:15` — test name `test_hf_local_tokenizer_preferred`

**Patterns.md references:**
- `.agents/patterns.md:56-57` — HuggingFace shard discovery and URL normalization gotchas

---

## Implementation Plan

### Phase 1: Delete HF Code from hub.py
*Focus: Remove all HF-specific methods and imports*

- [ ] 1.1 **Delete HF methods from `hub.py`**
  - Remove: `_hf_resolve_url()`, `_make_hf_store()`, `_get_hf_token()`,
    `_fetch_index_json()`, `_fetch_index_json_async()`,
    `_download_hf_file()`, `_download_hf_file_async()`,
    `_parse_shard_filenames()`
  - Remove: `_HF_BASE` constant
  - Remove: `HFDownloadError` class
  - Remove: `HTTPStore` from imports
  - File: `src/py/mogemma/hub.py`

- [ ] 1.2 **Update type signatures**
  - `_head_exists(store: LocalStore | HTTPStore, ...)` → `_head_exists(store: LocalStore | GCSStore, ...)`
  - (Or just `_head_exists(store, ...)` if both use the obstore protocol)
  - File: `src/py/mogemma/hub.py:81`

- [ ] 1.3 **Update error messages and docstrings**
  - Module docstring: `"Model resolution and GCS download helpers."`
  - Class docstring: `"Manages downloading and caching Gemma 4 models from Google Cloud Storage."`
  - `ModelNotFoundError` message: reference `gemma-data` bucket
  - `resolve_model()` strict error: mention GCS model ids
  - Tokenizer error: reference GCS download
  - File: `src/py/mogemma/hub.py` (multiple locations)

### Phase 2: Delete HF Test Files
*Focus: Remove HF-specific test files*

- [ ] 2.1 **Delete `src/py/tests/test_hf_download.py`**
  - This entire file tests HF-specific code (HTTPStore, HF_TOKEN, HF URLs)
  - Replaced by `test_gcs_download.py` from Chapter 1
  - File: delete `src/py/tests/test_hf_download.py`

- [ ] 2.2 **Delete `src/py/tests/test_hf_token.py`**
  - Tests HF_TOKEN env var and Bearer header — no longer relevant
  - File: delete `src/py/tests/test_hf_token.py`

- [ ] 2.3 **Update `test_tokenizer_resolution.py`**
  - Rename `test_hf_local_tokenizer_preferred` → `test_local_tokenizer_preferred`
  - Update any HF-specific comments
  - File: `src/py/tests/test_tokenizer_resolution.py:15`

### Phase 3: Update References
*Focus: Clean up remaining HF mentions across codebase*

- [ ] 3.1 **Update `model.py` docstrings**
  - `_detect_gemma4_variant()` docstring: remove "HuggingFace" prefix
  - File: `src/py/mogemma/model.py:110`

- [ ] 3.2 **Update `.agents/patterns.md` gotchas**
  - Remove: "HuggingFace Shard Discovery" gotcha (line 56)
  - Remove: "HuggingFace URL Normalization" gotcha (line 57)
  - Add: GCS-specific gotchas if any emerged during Chapters 1-2
  - File: `.agents/patterns.md:56-57`

- [ ] 3.3 **Update `README.md` if needed**
  - README already mentions GCS (`line 10`), verify no HF references remain
  - File: `README.md`

- [ ] 3.4 **Verify no `HF_TOKEN` references remain**
  - Grep entire codebase for `HF_TOKEN`, `huggingface`, `hf_` patterns
  - Should find zero matches outside of git history
  - Command: `rg -i 'hugging|hf_token|hf_base' src/`

### Verification

- [ ] `make lint` passes
- [ ] `make test` passes
- [ ] `grep -ri 'huggingface\|hf_token\|HTTPStore' src/` returns no matches
- [ ] `python -c "from obstore.store import HTTPStore"` is NOT imported by any mogemma module
- [ ] Full test suite passes with no HF-related test files
