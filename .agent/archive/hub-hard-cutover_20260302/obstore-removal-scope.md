# Obstore Removal Scope

## Objective

Define the exact deletion and rewrite scope required to remove active `obstore` usage from `HubManager.download()` during the hard cutover to the Rust+Mojo CFFI downloader.

## Current Active `obstore` Surface

Runtime `obstore` usage is currently isolated to [hub.py](/home/cody/code/c/mogemma/src/py/mogemma/hub.py):

1. Module import: `import obstore as obs`
2. Remote listing helper: `_list_remote_files(...)`
3. Tokenizer bucket-layout helper: `_get_tokenizer_path(...)`
4. Per-file download loop: `obs.get(...)`
5. Python concurrency wrapper: `ThreadPoolExecutor(max_workers=8)`

Repository-wide dependency references currently exist only in:

1. [pyproject.toml](/home/cody/code/c/mogemma/pyproject.toml)
2. `uv.lock`
3. [hub.py](/home/cody/code/c/mogemma/src/py/mogemma/hub.py)

No non-flow product docs currently reference `obstore`, so documentation scope is limited to migration/planning artifacts and any release notes added alongside the implementation PR.

## Keep vs Remove

Keep in the active Python hub path:

1. `resolve_model(...)` cached-directory short-circuit
2. Model ID normalization via `_clean_model_id(...)`
3. Cache-root normalization via `_cache_dir_for_model_id(...)`
4. Final cache-path safety validation before returning to callers
5. `_ensure_safetensors(...)`, but only after the native downloader reports a committed Orbax-backed result

Remove from the active Python hub path:

1. Direct `obstore` import and types
2. `_list_remote_files(...)`
3. `_get_tokenizer_path(...)` as bucket-layout policy embedded in Python
4. `obs.get(...)`-driven per-artifact transfer logic
5. `ThreadPoolExecutor`-based Python download scheduling

## Rewrite Scope by File

### [hub.py](/home/cody/code/c/mogemma/src/py/mogemma/hub.py)

Required rewrite:

1. Replace the current `download(...)` implementation with:
   - request assembly for the native downloader
   - a single Rust+Mojo CFFI invocation
   - response validation
   - post-commit `_ensure_safetensors(...)` gating
2. Validate that the returned cache directory resolves under `self.cache_path`
3. Validate required response fields before treating the download as committed
4. Keep `ModelNotFoundError` as the user-facing `NOT_FOUND` mapping

Required deletions:

1. `import obstore as obs`
2. `_list_remote_files(...)`
3. `_get_tokenizer_path(...)` from the active download path
4. Inline `_download_file(...)` helper
5. `ThreadPoolExecutor` usage in `download(...)`

### [model.py](/home/cody/code/c/mogemma/src/py/mogemma/model.py)

No direct `obstore` deletion is required here, but this file is part of the cutover contract:

1. Keep `_resolve_model_path(...)` calling `HubManager().resolve_model(..., download_if_missing=True, strict=True)`
2. Preserve the expectation that a committed model directory exposes `tokenizer.model` when the family requires a tokenizer
3. Do not introduce any model-layer fallback that reconstructs missing artifacts after hub download failure

### [test_hub.py](/home/cody/code/c/mogemma/src/py/tests/test_hub.py)

Tests to rewrite or add:

1. Red-first test proving `HubManager.download()` no longer calls `obstore` helpers
2. Red-first test proving the active path no longer uses a Python threadpool
3. Invalid returned-path rejection test
4. Partial-download cleanup test proving interrupted or failed native downloads do not leave a valid final cache directory
5. Integrity-failure cleanup test for checksum/format mismatch
6. Transport-failure mapping test
7. Local-filesystem-failure mapping test
8. Tokenizer-required behavior test based on downloader response rather than Python bucket inspection

Existing regressions to preserve:

1. Cached-path resolution remains a warm-cache fast path
2. Missing-model behavior still raises `HubManager.ModelNotFoundError`
3. `_ensure_safetensors(...)` still runs for valid committed cache trees

### [test_gemma_model.py](/home/cody/code/c/mogemma/src/py/tests/test_gemma_model.py)

Regression scope:

1. Keep `test_gemma_model_init_uses_hub_resolution`
2. Preserve strict rejection of unknown model IDs when hub resolution fails
3. Preserve tokenizer loading behavior from committed cache trees

No obstore-specific code should remain here after the cutover. This file should continue to assert caller-visible semantics only.

### [test_embeddings.py](/home/cody/code/c/mogemma/src/py/tests/test_embeddings.py)

Regression scope:

1. Keep `test_embedding_model_init_uses_hub_resolution`
2. Preserve tokenizer-required behavior for text embedding
3. Preserve pre-tokenized embedding path without adding new hub-side fallback behavior

No obstore-specific code should remain here after the cutover. This file should continue to assert caller-visible semantics only.

### [pyproject.toml](/home/cody/code/c/mogemma/pyproject.toml)

Dependency cleanup scope:

1. Remove `obstore` from main runtime dependencies once [hub.py](/home/cody/code/c/mogemma/src/py/mogemma/hub.py) no longer imports it
2. Remove `obstore` from test/build requirements once hub tests stop relying on it transitively
3. Regenerate `uv.lock` after dependency removal so lock state matches the manifest

## Deletion Order

The implementation PR should land deletions in this order:

1. Add or expose the native downloader contract used by Python hub integration
2. Rewrite [hub.py](/home/cody/code/c/mogemma/src/py/mogemma/hub.py) so the active download path no longer imports or calls `obstore`
3. Add red-first hub tests for no-`obstore`, no-threadpool, path validation, and cleanup semantics
4. Update regression tests in [test_gemma_model.py](/home/cody/code/c/mogemma/src/py/tests/test_gemma_model.py) and [test_embeddings.py](/home/cody/code/c/mogemma/src/py/tests/test_embeddings.py) only as needed to reflect the new committed-cache contract
5. Remove `obstore` from [pyproject.toml](/home/cody/code/c/mogemma/pyproject.toml)
6. Refresh `uv.lock`

This order avoids a mid-PR state where the manifest drops `obstore` before the runtime import is gone.

## No-Fallback Proof Conditions

The cutover is not complete unless all of the following are true:

1. [hub.py](/home/cody/code/c/mogemma/src/py/mogemma/hub.py) has no `obstore` import
2. `HubManager.download()` performs exactly one native downloader call on the cold path
3. No Python helper lists remote bucket contents or downloads individual artifacts
4. No feature flag, conditional import, or exception handler restores the old `obstore` path
5. [model.py](/home/cody/code/c/mogemma/src/py/mogemma/model.py) callers continue to use `HubManager.resolve_model(...)` without any alternate direct-download escape hatch

## Open Follow-On Work

Task `mogemma-h4n.3.2` should define the concrete exception taxonomy and cleanup behavior for:

1. `NOT_FOUND`
2. transport exhaustion or timeout
3. local filesystem failure
4. integrity failure
5. invalid returned cache paths
