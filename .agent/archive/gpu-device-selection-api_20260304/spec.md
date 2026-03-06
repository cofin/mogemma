# Flow: gpu-device-selection-api_20260304

## Flow Context

- **Flow ID:** `gpu-device-selection-api_20260304`
- **Status:** `planned`
- **Parent PRD:** `mojo-gpu-foundation_20260304`
- **PRD 1 Chapter:** 3 (device selection API and policy)
- **Scope:** Define and implement deterministic device selection semantics (`cpu`, `gpu`, `gpu:N`) with explicit capability detection and strict error behavior when requested GPUs are unavailable.

## Code Analysis Summary

### Files Examined

- `.agent/patterns.md`
- `.agent/research/gpu-acceleration-options_20260304/research.md`
- `src/py/mogemma/config.py`
- `src/py/mogemma/model.py`
- `src/mo/mogemma/core.mojo`
- `src/py/tests/test_embeddings.py`
- `src/py/tests/test_gemma_model.py`
- `src/py/tests/test_mojo_core.py`

### Key Findings (with file refs)

1. Device fields already exist in both configs but are not validated beyond generic numeric checks (`src/py/mogemma/config.py:20`, `src/py/mogemma/config.py:54`, `src/py/mogemma/config.py:72`).
2. Model initialization ignores device selection entirely and always initializes through `_core.init_model(metadata)` (`src/py/mogemma/model.py:95`, `src/py/mogemma/model.py:97`, `src/py/mogemma/model.py:300`).
3. Mojo `init_model_mojo` accepts only metadata and has no device parameter or capability hook (`src/mo/mogemma/core.mojo:274`, `src/mo/mogemma/core.mojo:606`).
4. Existing tests now assert parser validity, capability checks, and normalized device wiring through generation/embedding model init paths (`src/py/tests/test_backends.py`, `src/py/tests/test_embeddings.py`, `src/py/tests/test_gemma_model.py`).

## Relevant Patterns

1. **Hybrid Mojo-Python architecture:** API-level device semantics should be defined in Python and passed explicitly through the boundary (`.agent/patterns.md`).
2. **Deterministic validation:** device-selection behavior must be deterministic and testable (`.agent/patterns.md`).
3. **Contract testing:** add tests that validate device-policy contracts at boundaries (`.agent/patterns.md`).
4. **Research staging:** device API should land after backend abstraction and runtime-state seams are clear (`.agent/research/gpu-acceleration-options_20260304/research.md`).

## Requirements

### Functional Requirements

1. Define a normalized device grammar: `cpu`, `gpu`, `gpu:<index>`.
2. Add parser/validator that rejects malformed specs with actionable errors.
3. Define deterministic capability policy for unavailable GPU with explicit errors (no implicit downgrade path).
4. Thread normalized device selection through initialization path to backend/runtime layers.
5. Document behavior and examples for both embedding and generation APIs.

### Non-Functional Requirements

1. Preserve current `cpu` default behavior with no regressions.
2. Keep error messages stable and test-assertable.
3. Avoid hidden implicit downgrades; policy must be explicit in code and docs.
4. Ensure semantics are backend-agnostic for future `gpu_mojo` execution.

### Risks and Mitigations

1. **Risk:** ambiguous unavailable-GPU behavior creates production surprises.
   **Mitigation:** enforce strict error semantics and tests for capability branches.
2. **Risk:** parsing differences between Python and backend layers.
   **Mitigation:** normalize once in Python and pass structured selection downstream.
3. **Risk:** introducing device config changes breaks existing initialization flows.
   **Mitigation:** keep backward-compatible defaults and add migration tests.

## Implementation Plan

### Phase 1: API Contract and Policy Table

- [x] Define canonical device grammar and examples.
- [x] Define capability decision matrix for each requested device form.
- [x] Checkpoint: policy table approved and captured in flow spec/docs before coding.

### Phase 2: Validation and Normalization

- [ ] Implement config-level or model-init-level parsing/validation for device specs.
- [ ] Introduce normalized device descriptor passed through model initialization path.
- [ ] Checkpoint: invalid device strings fail fast with deterministic errors.

### Phase 3: Runtime Wiring

- [ ] Thread device descriptor to backend abstraction/runtime initialization seam.
- [ ] Add capability hooks for availability checks (CPU-only environments included).
- [ ] Checkpoint: selection result is visible in runtime state/debug surfaces.

### Phase 4: Documentation and Verification

- [ ] Update user-facing docs/examples for valid device strings and strict unavailable-GPU behavior.
- [ ] Add tests for parser, capability checks, and unavailable-GPU errors.
- [ ] **Manual verification task:** validate behavior on a CPU-only host (expected deterministic result) and with mocked GPU capability (expected selected device path).

## TDD-Oriented Task Checklist

- [ ] RED: add failing unit tests for valid/invalid device string parsing.
- [ ] RED: add failing tests for unavailable GPU behavior (strict error mode).
- [ ] RED: add integration tests proving normalized device data is carried into model/backend initialization.
- [ ] GREEN: implement parser + policy enforcement + runtime wiring.
- [ ] GREEN: update generation/embedding init paths to consume normalized selection.
- [ ] REFACTOR: centralize device parsing helpers to avoid duplicate logic between configs.
- [ ] REGRESSION: run `src/py/tests/test_gemma_model.py`, `src/py/tests/test_embeddings.py`, and target new device tests.
