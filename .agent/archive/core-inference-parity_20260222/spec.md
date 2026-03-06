# Flow: core-inference-parity_20260222

## Specification

### Goal

Replace placeholder Mojo core behavior with production-grade inference semantics so the release no longer depends on synthetic logits or random embedding outputs.

### Code Analysis Summary

1. `src/mo/core.mojo:18` returns `np.zeros(256000, dtype=np.float32)` in `step_mojo`, which is placeholder output.
2. `src/mo/core.mojo:29` returns `np.random.rand(batch_size, 768).astype(np.float32)` in `generate_embeddings_mojo`, which is placeholder output.
3. `src/py/mogemma/model.py:227` assumes `_core.step(...)` returns logits suitable for sampling, so placeholder logits directly affect generation behavior.
4. `src/py/mogemma/model.py:131` assumes `_core.generate_embeddings(...)` returns a real embedding matrix and validates only shape-level constraints.
5. `tools/hatch_build.py:20` builds `src/mo/core.mojo` into `src/py/mogemma/_core.so`, so runtime correctness depends on this file.
6. `src/py/tests/test_gemma_model.py` and `src/py/tests/test_embeddings.py` currently validate wrapper behavior mainly via stubs, not production core parity.

### User Decisions Applied

1. Release target is production-ready now, not staged alpha.
2. Scope includes full first-release readiness, even if broad.
3. Quality gates must be strict and non-soft-failing.

### Requirements

1. `init_model` must surface deterministic success/failure contract with actionable error messages.
2. `step` must return real backend logits, not synthetic arrays.
3. `generate_embeddings` must return model-generated embeddings, not random arrays.
4. Core output dtype and shape contracts must be explicit and enforced.
5. Python wrappers must remain compatible with the final core contract.
6. Tests must prove placeholder behavior is eliminated.

### Non-goals

1. Adding new user-facing model families beyond current Gemma scope.
2. Redesigning Python public API names.
3. Performance optimization tuning beyond correctness and contract stability.

## Implementation Plan

### Phase 1: Define and Lock Core Contracts

- [x] 1.1 Create a contract section in this flow documenting exact signatures, output dtypes, output shapes, and error conditions for `init_model`, `step`, and `generate_embeddings`.
- [x] 1.2 Define tokenizer/vocab assumptions required by `step` output shape and sampling pipeline.
- [x] 1.3 Define embedding output contract (`float32`, 2D matrix, row count equals batch size).
- [x] 1.4 Define error taxonomy for core failures (model not found, invalid model artifacts, backend runtime initialization failure).
- [x] 1.5 Define compatibility requirements for `src/py/mogemma/model.py` against finalized Mojo contract.

### Phase 2: Replace Placeholder Core Behavior

- [x] 2.1 Update `src/mo/core.mojo` `step_mojo` to invoke backend inference and return real logits.
- [x] 2.2 Update `src/mo/core.mojo` `generate_embeddings_mojo` to return real embeddings from backend path.
- [x] 2.3 Ensure `src/mo/core.mojo` `init_model_mojo` validates model path and surfaces explicit root-cause error.
- [x] 2.4 Remove debug/placeholder `print(...)` output from production paths unless gated behind explicit debug mode.
- [x] 2.5 Verify `PyInit__core` exports unchanged names (`init_model`, `generate_embeddings`, `step`) for Python compatibility.

### Phase 3: Align Python Wrapper Contracts

- [x] 3.1 Validate `src/py/mogemma/model.py` assumptions against final logits shape and adjust `_sample_next_token` preconditions if needed.
- [x] 3.2 Validate `src/py/mogemma/model.py` embedding conversion and row validation against finalized core output.
- [x] 3.3 Ensure RuntimeError paths remain explicit when core is absent or initialization fails.
- [x] 3.4 Add/adjust type hints and runtime checks for any changed core return signatures.
- [x] 3.5 Confirm no fallback path reintroduces synthetic behavior in Python.

### Phase 4: Add Core Parity Tests

- [x] 4.1 Add tests that fail if `step` returns constant/synthetic logits for distinct token inputs.
- [x] 4.2 Add tests that fail if embeddings are random placeholders disconnected from input/model behavior.
- [x] 4.3 Extend integration tests to assert contract-level properties from real core outputs.
- [x] 4.4 Add negative-path tests for invalid model initialization and verify surfaced messages.
- [x] 4.5 Ensure existing wrapper tests remain deterministic and maintain meaningful assertions.

### Phase 5: Validation and Evidence Capture

- [x] 5.1 Run `make build` and verify `_core.so` rebuilds cleanly.
- [x] 5.2 Run `make test` and ensure all tests pass.
- [x] 5.3 Run targeted integration tests for generation and embeddings after core changes.
- [x] 5.4 Record proof artifact in flow notes: commands run, commit hash, and test outcomes.
- [x] 5.5 Mark Beads tasks complete only after evidence is attached.

## Test Plan

1. `make build`
2. `uv run pytest src/py/tests/test_gemma_model.py -q`
3. `uv run pytest src/py/tests/test_embeddings.py -q`
4. `uv run pytest src/py/tests -q`
5. `make test`

## Validation Criteria

1. `src/mo/core.mojo` no longer contains synthetic logits/random embedding implementations.
2. Python generation and embedding paths consume real core outputs successfully.
3. Failure paths are explicit and reproducible.
4. All chapter-specific and suite-level tests pass.
5. Evidence is recorded in Beads and flow artifacts.

## Risks and Mitigations

1. Risk: Core API drift breaks Python wrappers.
   Mitigation: Lock contract first; add integration tests before finishing.
2. Risk: Non-deterministic runtime behavior causes flaky tests.
   Mitigation: Use deterministic test fixtures and contract-level assertions.
3. Risk: Build toolchain mismatch across environments.
   Mitigation: Validate via `make build` and CI matrix in later release chapters.
