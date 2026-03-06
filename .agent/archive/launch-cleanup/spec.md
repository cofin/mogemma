# Flow: Launch Cleanup

## Specification

### Code Analysis Summary

The codebase has completed the tokenizer migration (`transformers` → `tokenizers`) and partial Mojo consolidation, but accumulated dead code from iterative prototyping:

**Dead files/functions identified:**
- `vision_model.py` — stub returning `"A cat"`, imports duplicate `_core`
- `telemetry.py:20-31` — `trace_inference` decorator never called anywhere
- `core.mojo:27-46` — `generate_text_mojo` exported but never called from Python
- `core.mojo:5-10` — `process_image_mojo` only used by vision stub
- `model.py:221-224` — dummy fallback yields `"token_0 token_1..."` when `_core` is None
- `test_accuracy.mojo` — imports deleted `quant` module (broken)
- `test_multimodal_inference.mojo` — imports deleted `inference`, `cache`, `vision` modules (broken)
- `test_vision_bridge.mojo` — tests vision stub
- `test_vision_model.py` — tests vision stubs, incomplete assertions
- `scripts/benchmark.py` — referenced in git diff but file doesn't exist

**Config/dependency issues:**
- `pyproject.toml` — `mojo>=0.26.1` as required runtime dep (should be build-only)
- `pyproject.toml` — `[vision]` optional group for deleted feature
- `__init__.py` — exports `VisionGemmaModel`
- `config.py:24-33` — `EmbeddingConfig.__post_init__` silently passes on invalid paths

**Naming:**
- `GemmaModel` should be `SyncGemmaModel` for clarity alongside `AsyncGemmaModel`

### Requirements
1. Remove all vision-related code (files, exports, tests, Mojo functions, optional dep group)
2. Remove all broken Mojo test files that reference deleted modules
3. Remove unused `trace_inference` decorator and `generate_text_mojo` function
4. Replace generation dummy fallback with fail-fast RuntimeError
5. Remove `mojo` from required runtime dependencies
6. Rename `GemmaModel` → `SyncGemmaModel` across codebase
7. Update `__init__.py` lazy imports and `__all__`
8. Clean up `pyproject.toml` optional groups
9. Delete old `.agent/specs/` and reset `flows.md`

### Non-goals
- No new features or backends (ONNX, gemma.cpp, etc.)
- No changes to the embedding pipeline logic
- No changes to the Mojo `core.mojo` init/step/embeddings functions (only removing dead ones)

---

## Implementation Plan

### Phase 1: Dead Code Removal
- [ ] 1.1 Delete `src/py/mogemma/vision_model.py`
- [ ] 1.2 Delete `src/py/tests/test_vision_model.py`
- [ ] 1.3 Delete all 3 broken Mojo test files: `src/mo/tests/test_accuracy.mojo`, `test_multimodal_inference.mojo`, `test_vision_bridge.mojo`
- [ ] 1.4 Remove `process_image_mojo` and `generate_text_mojo` from `src/mo/core.mojo` (and their registrations in `PyInit__core`)
- [ ] 1.5 Remove the unused `trace_inference` decorator from `telemetry.py` (keep `tracer` and `_NoOpTracer` — those are used)
- [ ] 1.6 Remove `VisionGemmaModel` from `__init__.py` (`__all__`, `_EXPORT_TO_MODULE`, `_EXTRA_HINT`, `TYPE_CHECKING` imports)

### Phase 2: Runtime Hardening
- [ ] 2.1 Replace dummy fallback in `model.py:generate_stream` (lines 221-224) with `RuntimeError` matching the embedding path's error message pattern
- [ ] 2.2 Rename `GemmaModel` → `SyncGemmaModel` in `model.py`, `async_model.py`, `__init__.py`, `_EXPORT_TO_MODULE`, `_EXTRA_HINT`, `__all__`, all test files, and `smoke_test.py`
- [ ] 2.3 Run tests to confirm rename didn't break anything

### Phase 3: Packaging & Config
- [ ] 3.1 Remove `mojo>=0.26.1` from `[project] dependencies` in `pyproject.toml` (leave empty or just keep the package installable)
- [ ] 3.2 Remove `[vision]` optional dep group from `pyproject.toml`
- [ ] 3.3 Remove `vision` deps from `[all]` group
- [ ] 3.4 Clean `scripts/benchmark.py` phantom — check if it exists, remove reference or create the file
- [ ] 3.5 Reset `.agent/flows.md` to only reference this PRD
- [ ] 3.6 Run full test suite (`make test`) and lint (`make lint`) to validate

### Validation Criteria
- `make test` passes with 0 failures
- `make lint` passes (ruff check + format, mypy, pyright)
- No imports of deleted modules anywhere in codebase
- `python -c "import mogemma; print(dir(mogemma))"` shows no `VisionGemmaModel`
- `uv pip install -e .` works without requiring `mojo` at install time
