# Flow: seamless-model-delivery_20260222

## Specification

### Goal

Deliver zero-config model download behavior in the default install by making Hugging Face client support part of base runtime and hardening local/cache/remote resolution behavior.

### Code Analysis Summary

1. `src/py/mogemma/hub.py:26` resolves local path and cache path, then optionally downloads based on `download_if_missing`.
2. `src/py/mogemma/hub.py:51` raises if `snapshot_download` is missing.
3. `src/py/mogemma/model.py:103` and `src/py/mogemma/model.py:180` call `resolve_model(..., download_if_missing=True)` by default.
4. `src/py/mogemma/_typing.py:22` conditionally imports `huggingface_hub.snapshot_download`, exposing optional dependency state.
5. `pyproject.toml` currently keeps `huggingface-hub` in optional `hub` extra instead of base dependencies.

### Requirements

1. Base install must include required HF client dependency for download path.
2. Model resolution order must remain local path -> cache -> remote download.
3. Offline and restricted-access failure behavior must be explicit and actionable.
4. Hub/cache behavior must be covered by deterministic tests.
5. Public docs must reflect no-config default download behavior.

### Non-goals

1. Implementing a full non-HF downloader stack for alternative registries.
2. Changing public model constructor signatures.
3. Adding heavy background download managers.

## Implementation Plan

### Phase 1: Dependency Policy Migration

- [x] 1.1 Move `huggingface-hub` from optional extras to base `[project.dependencies]` in `pyproject.toml`.
- [x] 1.2 Keep optional extras coherent after migration (remove redundant `hub` or convert it to compatibility alias policy).
- [x] 1.3 Regenerate lock state to reflect dependency policy.
- [x] 1.4 Ensure package metadata and install docs remain consistent with new policy.

### Phase 2: Hub Manager Behavior Hardening

- [x] 2.1 Validate `_is_hf_model_id` heuristic against expected local-path edge cases.
- [x] 2.2 Ensure `resolve_model` behavior is deterministic for local, cache-hit, and cache-miss scenarios.
- [x] 2.3 Add explicit handling/messages for auth-required repos and offline mode.
- [x] 2.4 Ensure `download` forwards relevant kwargs safely and predictably.
- [x] 2.5 Ensure cache directory behavior is stable and documented.

### Phase 3: Typing Facade Consistency

- [x] 3.1 Keep `src/py/mogemma/_typing.py` and `src/py/mogemma/typing.py` as the single import facade for runtime optional integrations.
- [x] 3.2 Reconcile facade status constants with new base dependency policy.
- [x] 3.3 Remove dead optional-branch handling that is no longer reachable under base install policy.

### Phase 4: Test Coverage Expansion

- [x] 4.1 Expand `src/py/tests/test_hub.py` for full resolution matrix: local path, cache hit, cache miss with download, invalid model IDs.
- [x] 4.2 Add tests for download failure paths (network failure, auth failure) using mocks.
- [x] 4.3 Add tests ensuring model constructors trigger expected resolution paths.
- [x] 4.4 Keep tests deterministic and network-independent.

### Phase 5: Documentation and DX

- [x] 5.1 Update README installation/dependency tables to reflect default HF download support.
- [x] 5.2 Add concise troubleshooting section for auth/offline/caching.
- [x] 5.3 Ensure examples for HF IDs and local paths both execute and match actual behavior.
- [x] 5.4 Verify wording stays concise and non-promotional.

### Phase 6: Validation and Evidence

- [x] 6.1 Run targeted hub/delivery tests.
- [x] 6.2 Run full test suite and lint gate after dependency policy change.
- [x] 6.3 Record evidence for install behavior and model resolution scenarios.

## Test Plan

1. `uv run pytest src/py/tests/test_hub.py -q`
2. `uv run pytest src/py/tests/test_gemma_model.py -q`
3. `uv run pytest src/py/tests/test_embeddings.py -q`
4. `uv run pytest src/py/tests -q`
5. `make lint`
6. `make test`

## Validation Criteria

1. Default install includes HF client dependency and supports download without extra configuration.
2. Resolution behavior is deterministic across local/cache/remote paths.
3. Offline/auth errors are explicit and actionable.
4. Tests prove delivery paths and failure modes.
5. Docs reflect real behavior and dependency policy.

## Risks and Mitigations

1. Risk: Dependency migration changes environment reproducibility.
   Mitigation: Lockfile refresh and install-validation tests in clean environments.
2. Risk: Heuristic misclassifies model IDs vs local paths.
   Mitigation: Add explicit edge-case tests and path normalization checks.
