# Flow: hub-hard-cutover_20260302

## Specification

### Goal

Plan strict migration of `HubManager.download()` away from obstore and into the Rust+Mojo CFFI path, with no backward compatibility fallback.

### Code Analysis Summary

Files analyzed:

- `src/py/mogemma/hub.py`
- `src/py/mogemma/model.py`
- `src/py/tests/test_hub.py`
- `src/py/tests/test_gemma_model.py`
- `src/py/tests/test_embeddings.py`
- `.agent/specs/hub-hard-cutover_20260302/hub-cffi-integration-map.md`

Key findings:

1. `resolve_model()` is the stable entrypoint used by both generation and embedding models, so any cutover bug will immediately affect model initialization across both APIs.
2. `download()` currently creates the final cache directory before the remote object list is validated, which makes partial cache trees possible on failure.
3. `_get_tokenizer_path()` and the Python artifact loop currently encode bucket-layout knowledge that must disappear from the active path after cutover.
4. The existing tests focus on successful resolution and strict missing-model behavior, but they do not yet require atomic staging cleanup, path validation, or stable mapped downloader error categories.

### Requirements

1. Remove active obstore list/get/store usage from the download path and ensure no runtime fallback remains behind feature flags or conditional imports.
2. Preserve current cache layout and model-id normalization semantics unless a revision explicitly changes them.
3. Define strict failure handling for missing model, partial download, integrity failure, local disk failure, and transport errors.
4. Define deletion and cleanup scope in code/tests/dependencies.
5. Keep the Python-side download path thin enough that performance improvements from the Rust downloader are not lost to Python bookkeeping.

### Cutover Contract

1. `HubManager.resolve_model()` keeps its current cached-path short-circuit behavior and still returns a `Path`.
2. `HubManager.download()` becomes a request assembly, CFFI invocation, response validation, and post-commit normalization function; it is not allowed to iterate remote files or open remote streams directly.
3. The Python layer may validate that the returned `cache_dir` is inside `self.cache_path` and that required response fields are present, but it must not reconstruct downloader state by rescanning or rewriting the full artifact set.
4. `_ensure_safetensors()` is called only after the downloader reports a committed cache tree and only when the response marks the result as Orbax-backed.
5. Missing tokenizer or required model artifacts must fail closed with deterministic exceptions; silent omission is not acceptable.

### Required File-Level Touch Points

The implementation PR for this flow must explicitly account for these locations:

1. `src/py/mogemma/hub.py`: remove obstore imports, remove `_list_remote_files()`, replace download loop with one downloader call, and harden local path validation.
2. `src/py/mogemma/model.py`: preserve `resolve_model(... download_if_missing=True, strict=True)` behavior and ensure no downstream code depends on obstore-specific artifacts.
3. `src/py/tests/test_hub.py`: add cutover-specific tests for atomic cleanup, mapped downloader failures, tokenizer expectations, and cache-path safety.
4. `src/py/tests/test_gemma_model.py` and `src/py/tests/test_embeddings.py`: confirm initialization still resolves through `HubManager` and preserves current caller-visible semantics.
5. `pyproject.toml`: remove or demote obstore dependency only when no active runtime path still imports it.

### Failure-Mode Requirements

1. If the downloader reports `NOT_FOUND`, `HubManager.ModelNotFoundError` remains the user-facing outcome.
2. Transport exhaustion, retry exhaustion, or timeouts must raise a deterministic connection-style failure with model context and no partial final cache directory.
3. Integrity failures must leave no committed final cache tree and must surface as non-retryable terminal failures.
4. Local filesystem failures must surface distinctly from upstream transport failures so operators know whether retrying is useful.
5. Returned paths must be normalized and validated so the Python layer cannot be tricked into accepting a path outside the configured cache root.

### Performance Guardrails

1. No `ThreadPoolExecutor` or equivalent Python-side per-artifact scheduling remains in `hub.py` after cutover.
2. Python-side download work is limited to request construction, one downloader call, result validation, and optional post-commit conversion.
3. The implementation must avoid repeated filesystem traversal of the committed artifact tree in the success path.
4. Warm-cache behavior must remain a fast local check with zero downloader/network work.
5. Any downloader/session/client setup cost must be paid once per `HubManager.download()` call; repeated native-handle creation per file blocks closure of this flow.

## Planning Artifacts

The following planning artifacts now define the detailed cutover contract for this flow:

1. [hub-cffi-integration-map.md](/home/cody/code/c/mogemma/.agent/specs/hub-hard-cutover_20260302/hub-cffi-integration-map.md): maps current `hub.py` integration points to the native downloader path.
2. [obstore-removal-scope.md](/home/cody/code/c/mogemma/.agent/specs/hub-hard-cutover_20260302/obstore-removal-scope.md): enumerates code, test, and dependency deletions required to remove active `obstore` usage.
3. [hard-cutover-failure-semantics.md](/home/cody/code/c/mogemma/.agent/specs/hub-hard-cutover_20260302/hard-cutover-failure-semantics.md): defines deterministic exception mapping and cleanup behavior for downloader failures.
4. [tokenizer-artifact-handoff.md](/home/cody/code/c/mogemma/.agent/specs/hub-hard-cutover_20260302/tokenizer-artifact-handoff.md): defines the Python-to-native request/response contract and tokenizer/artifact ownership boundary.
5. [chapter4-gate-inputs.md](/home/cody/code/c/mogemma/.agent/specs/hub-hard-cutover_20260302/chapter4-gate-inputs.md): defines benchmark coverage, fault-injection coverage, and release blockers for the cutover.

These artifacts are normative for implementation planning and should be treated as the detailed acceptance contract for the eventual code PR.

## Implementation Plan

### Phase 1: Integration planning

- [ ] 1.1 `mogemma-h4n.3.1` Map the exact integration points in `src/py/mogemma/hub.py`, including cache-dir creation, error mapping, and `_ensure_safetensors()` trigger points that must survive the cutover.
- [ ] 1.2 Checkpoint: define tokenizer/model artifact handoff from the hub layer to the Mojo boundary, including which fields are requested, which are returned, and which invariants Python must validate.

Phase 1 is incomplete unless it is obvious which existing helper functions disappear, which remain, and which tests will fail first when the cutover starts.

### Phase 2: Hard cutover planning

- [ ] 2.1 `mogemma-h4n.3.3` Define obstore removal scope across code, tests, docs, and dependencies, and make the deletion order explicit so import breakage does not happen mid-PR.
- [ ] 2.2 `mogemma-h4n.3.2` Define hard-cutover failure semantics for missing model, partial download, transport failures, checksum mismatch, and invalid returned paths.

Phase 2 must leave no ambiguity about atomic staging cleanup, what happens to partially downloaded artifacts after interruption, or which exceptions callers should assert on.

Phase 2 deliverables are captured in:

1. [obstore-removal-scope.md](/home/cody/code/c/mogemma/.agent/specs/hub-hard-cutover_20260302/obstore-removal-scope.md)
2. [hard-cutover-failure-semantics.md](/home/cody/code/c/mogemma/.agent/specs/hub-hard-cutover_20260302/hard-cutover-failure-semantics.md)
3. [tokenizer-artifact-handoff.md](/home/cody/code/c/mogemma/.agent/specs/hub-hard-cutover_20260302/tokenizer-artifact-handoff.md)

### Phase 3: Release readiness

- [ ] 3.1 Produce Chapter 4 gate inputs: required benchmark scenarios, required failure injections, and release-blocking signals tied to the new downloader path.
- [ ] 3.2 Track deferred safetensors improvements separately and keep them out of the initial hard cutover scope unless they are required to preserve current caller-visible behavior.

Phase 3 gate planning is captured in [chapter4-gate-inputs.md](/home/cody/code/c/mogemma/.agent/specs/hub-hard-cutover_20260302/chapter4-gate-inputs.md).

## TDD-Oriented Task Checklist

- [ ] RED: add failing hub tests that prove the active download path no longer calls obstore helpers or Python threadpool transfer code.
- [ ] RED: add failing tests for invalid returned paths, interrupted-download cleanup, and integrity-failure cleanup semantics.
- [ ] GREEN: wire the new downloader response path into `HubManager.download()` until the cutover tests pass.
- [ ] REFACTOR: remove obsolete obstore helpers/imports while keeping cache normalization and strict-mode behavior readable.
- [ ] REGRESSION: rerun `src/py/tests/test_hub.py`, `src/py/tests/test_gemma_model.py`, and `src/py/tests/test_embeddings.py`.

### Required Validation Evidence

1. Updated hub integration map showing the final Python entrypoints and the removed obstore helpers.
2. A removal-scope artifact naming the exact code, test, and dependency deletions required for the hard cutover.
3. A handoff contract naming the exact Python request fields, native response fields, and tokenizer/artifact invariants.
4. An error-mapping table tying downloader categories to Python exception types and messages.
5. A cleanup contract proving partial staging state is not considered a valid cache hit on rerun.
6. A gate-input artifact naming the benchmark scenarios, required fault injections, and release-blocking signals for the new downloader path.

## Task Acceptance Criteria

### `mogemma-h4n.3.1` Map hub integration points

1. Lists every obstore call site currently used by `HubManager.download()`.
2. Maps each call site to the replacement Mojo CFFI invocation or local validation step.
3. Documents required pre/post conditions at each integration point, including cache path safety and `_ensure_safetensors()` trigger rules.

### Tokenizer/artifact handoff checkpoint

1. Defines the exact data passed from the Python hub layer into the Mojo boundary.
2. Defines returned values and local path guarantees expected by callers.
3. Defines strict handling for missing tokenizer artifacts and for families that intentionally do not require a tokenizer.

### `mogemma-h4n.3.3` Obstore/dependency removal scope

1. Enumerates files/imports/tests to remove or rewrite for active download path.
2. Identifies package/dependency manifest changes required by cutover and the order they can land safely.
3. Confirms no runtime fallback path to obstore remains in download flow.

### `mogemma-h4n.3.2` Hard-cutover failure semantics

1. Defines deterministic error behavior for not found, partial download, transport failures, local disk failures, and invalid returned paths.
2. Defines integrity failure behavior (checksum/format mismatch) with explicit abort and cleanup semantics.
3. Ensures failure modes are surfaced to callers without silent recovery paths or partial final cache trees.
