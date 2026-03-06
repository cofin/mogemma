# Flow: gpu-runtime-state-refactor_20260304

## Flow Context

- **Flow ID:** `gpu-runtime-state-refactor_20260304`
- **Status:** `planned`
- **Parent PRD:** `mojo-gpu-foundation_20260304`
- **PRD 1 Chapter:** 2 (runtime/session-state refactor)
- **Scope:** Remove rebuild-heavy token-step behavior, establish reusable session state, and split host/device allocation concerns for later GPU execution.

## Code Analysis Summary

### Files Examined

- `.agent/patterns.md`
- `.agent/research/gpu-acceleration-options_20260304/research.md`
- `src/py/mogemma/model.py`
- `src/mo/mogemma/core.mojo`
- `src/mo/mogemma/layers.mojo`
- `src/mo/mogemma/model.mojo`
- `src/py/tests/test_mojo_core.py`
- `src/mo/tests/test_altup_contract.mojo`
- `src/mo/tests/test_mojo.py`

### Key Findings (with file refs)

1. Runtime dictionary built in `init_model_mojo` stores metadata and cache arrays, but token-step rebuild still occurs on each call (`src/mo/mogemma/core.mojo:350`, `src/mo/mogemma/core.mojo:362`, `src/mo/mogemma/core.mojo:366`).
2. `step_mojo` reconstructs `ModelWeights`/`NanoModelWeights` from runtime metadata every token (`src/mo/mogemma/core.mojo:411`, `src/mo/mogemma/core.mojo:436`), matching the known rebuild-overhead warning.
3. `generate_embeddings_mojo` also performs fresh RoPE/KV allocations and model construction per invocation (`src/mo/mogemma/core.mojo:494`, `src/mo/mogemma/core.mojo:506`, `src/mo/mogemma/core.mojo:524`, `src/mo/mogemma/core.mojo:527`).
4. Position state is mutable (`llm["pos"]`) and reset behavior is partially managed in Python (`src/mo/mogemma/core.mojo:459`, `src/py/mogemma/model.py:342`), so session lifecycle semantics need explicit ownership.
5. Nano path correctness depends on KV-sharing and specialized layer behavior; refactor cannot alter those semantics (`src/mo/mogemma/core.mojo:261`, `src/mo/mogemma/layers.mojo:842`, `src/mo/mogemma/layers.mojo:907`).

## Relevant Patterns

1. **Model rebuild overhead:** runtime descriptors should be cached and reused in hot paths (`.agent/patterns.md`).
2. **Gemma3 Nano constraints:** KV sharing and weighted RMSNorm semantics are parity-critical (`.agent/patterns.md`).
3. **Contract testing:** preserve cross-language behavior via contract-style checks (`.agent/patterns.md`, `src/mo/tests/test_altup_contract.mojo:21`).
4. **Deterministic validation:** parity checks should use deterministic settings and fixture-driven assertions (`.agent/patterns.md`).

## Requirements

### Functional Requirements

1. Introduce explicit runtime/session structures that hold pre-built model-weight views for standard and nano paths.
2. Ensure token stepping reuses prepared model state and cache buffers without per-token reconstruction.
3. Separate long-lived session allocations from transient per-call scratch allocations.
4. Define and implement reset semantics for `pos` and KV state across generation sessions.
5. Preserve nano KV-share behavior and layer semantics while refactoring internals.

### Non-Functional Requirements

1. No regression in current CPU correctness tests.
2. Measurable reduction in per-token overhead versus current rebuild-heavy baseline.
3. Memory lifetime safety across Python loader pointers and Mojo runtime structures.
4. Refactor should be backend-ready for Chapter 1 and Chapter 3 integration.

### Risks and Mitigations

1. **Risk:** stale pointers or invalid ownership when caching model structures.
   **Mitigation:** define clear lifecycle and add explicit teardown/reset tests.
2. **Risk:** subtle nano parity regressions from cache/write-path changes.
   **Mitigation:** add deterministic nano parity checks before and after refactor.
3. **Risk:** session reset bugs cause cross-request leakage.
   **Mitigation:** require tests for consecutive generations and explicit reset behavior.

## Implementation Plan

### Phase 1: Baseline and State Model

- [x] Capture current state transitions (`init_model` -> `step` -> `generate_embeddings`) and document allocation boundaries.
- [x] Define runtime/session structs for cached model views and mutable generation state.
- [x] Checkpoint: design notes map each current mutable field (`pos`, caches, runtime metadata) to new owner.

### Phase 2: Hot-Path Refactor

- [x] Remove per-token model reconstruction from `step_mojo` by caching prepared model structures.
- [x] Rework embedding path to reuse reusable runtime components while isolating per-input scratch.
- [x] Checkpoint: instrumentation or counters confirm model-view build count is no longer per token.

### Phase 3: Session Semantics and Safety

- [x] Add/reset APIs or lifecycle hooks for deterministic session reuse.
- [x] Validate nano KV-share invariants after refactor.
- [x] Checkpoint: consecutive generation and embedding operations show stable correctness.

### Phase 4: Validation and Evidence

- [x] Extend tests for runtime reuse, reset, and no-regression behavior.
- [x] Record perf evidence (token-step overhead before/after).
- [x] **Manual verification task:** run a controlled multi-prompt generation sequence against one loaded model and verify session reuse occurs without correctness drift.

## TDD-Oriented Task Checklist

- [x] RED: add failing tests that assert model-view construction does not happen per token in `step`.
- [x] RED: add failing tests for session reset and consecutive-generation state isolation.
- [x] RED: add failing nano-focused regression checks around KV-share-dependent behavior.
- [x] GREEN: implement cached runtime/session model and hot-path refactor in Mojo core.
- [x] GREEN: wire Python entrypoints to the new session semantics.
- [x] REFACTOR: separate allocation helpers for long-lived caches vs transient scratch buffers.
- [x] REGRESSION: run `src/py/tests/test_mojo_core.py` and relevant Mojo contract suites (`src/mo/tests/test_altup_contract.mojo`, stable subset via `src/mo/tests/test_mojo.py`).
