# Flow: gpu-backend-abstraction_20260304

## Flow Context

- **Flow ID:** `gpu-backend-abstraction_20260304`
- **Status:** `planned`
- **Parent PRD:** `mojo-gpu-foundation_20260304`
- **PRD 1 Chapter:** 1 (backend contract and dispatch seam)
- **Scope:** Define backend abstractions between Python API surface, runtime/session state, and Mojo kernel execution paths.

## Code Analysis Summary

### Files Examined

- `.agent/patterns.md`
- `.agent/research/gpu-acceleration-options_20260304/research.md`
- `src/py/mogemma/config.py`
- `src/py/mogemma/model.py`
- `src/py/mogemma/loader.py`
- `src/mo/mogemma/core.mojo`
- `src/mo/mogemma/layers.mojo`
- `src/mo/mogemma/model.mojo`
- `src/py/tests/test_gemma_model.py`
- `src/py/tests/test_embeddings.py`
- `src/py/tests/test_mojo_core.py`

### Key Findings (with file refs)

1. Python runtime is hard-wired to one extension backend (`mogemma._core`) with direct calls for `init_model`, `step`, and `generate_embeddings` (`src/py/mogemma/model.py:19`, `src/py/mogemma/model.py:97`, `src/py/mogemma/model.py:226`, `src/py/mogemma/model.py:345`, `src/py/mogemma/model.py:355`).
2. Mojo extension exports a single engine surface (`init_model`, `generate_embeddings`, `step`) without a backend-selection API (`src/mo/mogemma/core.mojo:606`, `src/mo/mogemma/core.mojo:607`, `src/mo/mogemma/core.mojo:608`).
3. Runtime build logic and kernel dispatch are currently intertwined inside `core.mojo`, which makes adding multiple backends risky without an explicit abstraction seam (`src/mo/mogemma/core.mojo:274`, `src/mo/mogemma/core.mojo:366`).
4. Existing Python tests assert API behavior around generation/embedding but not backend selection semantics, leaving a coverage gap for pluggable backend routing (`src/py/tests/test_gemma_model.py:68`, `src/py/tests/test_embeddings.py:67`).

## Relevant Patterns

1. **Hybrid Mojo-Python boundary:** keep Python API stable while changing engine internals (`.agent/patterns.md`).
2. **Contract testing:** preserve cross-boundary guarantees with explicit contract tests (`.agent/patterns.md`, `src/mo/tests/test_altup_contract.mojo:21`).
3. **Deterministic validation:** use deterministic decoding settings for parity and regression tests (`.agent/patterns.md`).
4. **Model rebuild overhead warning:** architecture should avoid rebuild-heavy hot paths (`.agent/patterns.md`, `.agent/research/gpu-acceleration-options_20260304/research.md`).

## Requirements

### Functional Requirements

1. Introduce a backend interface for model initialization, token stepping, and embedding generation.
2. Add backend dispatch entry points supporting `cpu_mojo` immediately, with defined extension points for `gpu_mojo` and optional adapters.
3. Preserve `SyncGemmaModel` / `EmbeddingModel` public usage and defaults while routing through the new abstraction.
4. Ensure backend selection is explicit in runtime state so later flows can attach device semantics safely.

### Non-Functional Requirements

1. No behavior regression for current CPU path and existing tests.
2. Clear ownership boundaries between Python orchestration and Mojo execution.
3. Low overhead dispatch (no per-token backend re-resolution).
4. Backward-compatible API docs and import surface.

### Risks and Mitigations

1. **Risk:** abstraction leaks backend-specific assumptions into high-level API.
   **Mitigation:** define strict backend protocol and shared request/response contracts first.
2. **Risk:** introducing abstraction breaks existing tests silently.
   **Mitigation:** require red/green coverage updates before implementation completion.
3. **Risk:** future GPU backend blocked by missing runtime/session seam.
   **Mitigation:** keep backend interface separate from runtime state objects (Chapter 2 dependency).

## Implementation Plan

### Phase 1: Backend Contract Design

- [x] Define backend protocol types and runtime data contracts in Python (`init`, `step`, `embed`).
- [x] Define canonical backend identifiers and resolution order.
- [x] Checkpoint: architecture decision record notes why `cpu_mojo` is the initial default backend.

### Phase 2: Python Dispatch Integration

- [x] Refactor `SyncGemmaModel` and `EmbeddingModel` to call backend abstraction instead of direct `_core` calls.
- [x] Keep `_core` path behind a concrete `cpu_mojo` backend adapter.
- [ ] Checkpoint: all existing CPU model/embedding tests pass through abstraction layer.

### Phase 3: Mojo Boundary Alignment

- [ ] Align `_core` entrypoint expectations with backend abstraction contract.
- [ ] Document required future hooks for GPU-capable backend module(s) without implementing kernels yet.
- [ ] Checkpoint: scoped integration notes identify exact hand-off points to Chapter 2 and Chapter 3.

### Phase 4: Validation and Planning Artifacts

- [ ] Add/extend tests to assert backend routing behavior and default backend consistency.
- [ ] Document backend architecture and migration strategy in flow notes.
- [ ] **Manual verification task:** run a local generation + embedding smoke workflow with default settings and verify logs/inspection prove `cpu_mojo` dispatch path is used end-to-end.

## TDD-Oriented Task Checklist

- [ ] RED: add failing unit tests for backend resolution defaults and invalid backend IDs.
- [ ] RED: add failing tests proving `SyncGemmaModel` and `EmbeddingModel` route through backend abstraction (not direct `_core` calls).
- [ ] GREEN: implement minimal `cpu_mojo` backend adapter and dispatcher.
- [ ] GREEN: update model classes to use dispatcher while preserving output contracts.
- [ ] REFACTOR: consolidate shared backend call plumbing and error translation.
- [ ] REGRESSION: run `src/py/tests/test_gemma_model.py`, `src/py/tests/test_embeddings.py`, and targeted `src/py/tests/test_mojo_core.py`.
