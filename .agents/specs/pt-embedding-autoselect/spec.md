# Flow: pt-embedding-autoselect

## Specification

### Goal
Change `EmbeddingConfig` to default to pre-trained (`-pt`) model variant instead of
instruction-tuned (`-it`). Pre-trained models produce higher-quality embeddings because
IT fine-tuning optimizes for instruction-following, not representation learning.

### Depends On
- Chapter 1 (`gcs-download-backend`) — GCS download must support `-pt` variants

### Code Analysis Summary

**Files to modify:**
- `src/py/mogemma/config.py:19` — change `EmbeddingConfig.model_path` default
- `src/py/tests/test_config.py:33-35` — update default model path assertion

**Files to potentially modify:**
- `src/py/mogemma/model.py` — `_instruction_tuned` flag detection (currently based on `-it` in name)
- `src/py/mogemma/hub.py:64-67` — `_clean_model_id()` must handle `-pt` suffix correctly

**Current state:**
- `EmbeddingConfig.model_path` = `"google/gemma-4-26B-A4B-it"` (`config.py:19`)
- `GenerationConfig.model_path` = `"google/gemma-4-26B-A4B-it"` (`config.py:62`)
- `_clean_model_id("google/gemma-4-26B-A4B-pt")` → `"gemma4-26B-A4B-pt"` (works correctly)
- GCS path would be: `checkpoints/gemma4-26b-a4b-pt/` (needs lowercase)

**Test that references the default:**
- `test_config.py:33-35`: `TestEmbeddingConfig.test_default_model_path` asserts `"google/gemma-4-26B-A4B-it"`

### Requirements

1. `EmbeddingConfig.model_path` defaults to `"google/gemma-4-26B-A4B-pt"`
2. `GenerationConfig.model_path` remains `"google/gemma-4-26B-A4B-it"` (unchanged)
3. `_clean_model_id()` handles `-pt` suffix same as `-it` (already works)
4. GCS path mapping produces correct lowercase: `checkpoints/gemma4-26b-a4b-pt/`
5. Docstrings updated to explain the PT default for embeddings

---

## Implementation Plan

### Phase 1: Config Change
*Focus: Change defaults and update documentation*

- [ ] 1.1 **Update `EmbeddingConfig.model_path` default**
  - Change: `"google/gemma-4-26B-A4B-it"` → `"google/gemma-4-26B-A4B-pt"`
  - Update docstring to explain: "Pre-trained variant used by default for embeddings
    (IT models are optimized for instruction-following, not representation quality)"
  - File: `src/py/mogemma/config.py:19-20`

- [ ] 1.2 **Update test assertion**
  - Change: `assert str(config.model_path) == "google/gemma-4-26B-A4B-it"`
    → `assert str(config.model_path) == "google/gemma-4-26B-A4B-pt"`
  - File: `src/py/tests/test_config.py:33-35`

### Phase 2: Validate Integration
*Focus: Ensure PT models work correctly through the full pipeline*

- [ ] 2.1 **Verify `_clean_model_id()` handles PT correctly**
  - Confirm: `_clean_model_id("google/gemma-4-26B-A4B-pt")` → `"gemma4-26B-A4B-pt"`
  - Lowercase for GCS: `"gemma4-26b-a4b-pt"` → `checkpoints/gemma4-26b-a4b-pt/`
  - Add test case if not already present
  - File: `src/py/tests/test_hub.py`

- [ ] 2.2 **Verify `_instruction_tuned` flag**
  - In `model.py`, the generation model checks for `-it` in model name to enable
    turn formatting. Verify PT models correctly set `_instruction_tuned = False`
  - File: `src/py/mogemma/model.py` (read-only verification)

- [ ] 2.3 **Verify embedding model initialization with PT**
  - PT models should have identical architecture to IT (same config.json fields)
  - The only difference is weight values, not structure
  - Verify `_detect_gemma4_variant()` works on PT models (it should — reads config.json)

### Verification

- [ ] `make lint` passes
- [ ] `make test` passes
- [ ] `EmbeddingConfig().model_path` returns `"google/gemma-4-26B-A4B-pt"`
- [ ] `GenerationConfig().model_path` still returns `"google/gemma-4-26B-A4B-it"`
