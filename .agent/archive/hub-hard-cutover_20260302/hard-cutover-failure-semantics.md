# Hard-Cutover Failure Semantics

## Objective

Define deterministic caller-visible behavior for `HubManager.download()` once the active path uses the Rust+Mojo CFFI downloader and no `obstore` fallback remains.

## Contract Summary

Cold-path downloads must follow this rule set:

1. Python performs one native downloader call
2. Python validates the returned cache path and required response fields
3. Python treats the result as committed only after path validation passes
4. Any failure before commit leaves no final cache directory
5. Any integrity failure discovered after commit must delete the committed directory before surfacing the exception

Warm-cache behavior is unchanged:

1. `resolve_model(...)` returns a valid local cache directory immediately
2. No downloader work occurs on a valid warm-cache hit

## Error Mapping Table

| Downloader / validation category | Python exception | Retryable | Required cleanup | Message contract |
| --- | --- | --- | --- | --- |
| model prefix missing / `NOT_FOUND` | `HubManager.ModelNotFoundError` | No | remove staging if created; leave existing warm cache untouched | `Model '<clean_id>' was not found in the public gemma-data bucket.` |
| transport timeout / retry exhaustion / interrupted remote transfer before commit | `ConnectionError` | Yes | remove staging; do not create final cache dir | `Download failed for '<clean_id>': transport error (<detail>)` |
| local mkdir/write/fsync/rename failure | `OSError` | Depends on operator action | remove staging when possible; never treat partial output as cache hit | `Download failed for '<clean_id>': local filesystem error (<detail>)` |
| checksum mismatch / format mismatch / missing required artifact in committed result | `ValueError` | No | remove staging or delete committed final cache dir before raising | `Download failed for '<clean_id>': integrity error (<detail>)` |
| invalid native return path / path escapes cache root / malformed result payload | `ValueError` | No | delete staging or committed dir associated with invalid result | `Downloader returned invalid cache path for '<clean_id>'` |

## Failure Semantics by Scenario

### 1. Model not found

Trigger:

1. Native downloader reports no checkpoint prefix for the normalized model ID

Behavior:

1. Raise `HubManager.ModelNotFoundError`
2. Preserve the existing user-facing not-found message
3. Do not create or keep a final cache directory
4. Do not fall back to any Python bucket listing or alternate transport path

### 2. Partial download or interrupted transfer

Trigger:

1. Timeout
2. Retry exhaustion
3. Connection reset
4. Downloader abort before commit

Behavior:

1. Raise `ConnectionError`
2. Include model context and native failure detail in the message
3. Remove all staging artifacts before returning control to the caller
4. Ensure rerunning `resolve_model(..., download_if_missing=True, strict=True)` does not treat leftover partial files as a cache hit

### 3. Local filesystem failure

Trigger:

1. Cache directory creation failure
2. Write failure
3. `fsync` failure
4. Atomic rename/commit failure
5. Cleanup failure after a native error

Behavior:

1. Raise `OSError`
2. Preserve the underlying filesystem detail when possible
3. Never mask local I/O failure as a transport problem
4. Remove staging artifacts when possible; if cleanup also fails, surface the original failure and log cleanup failure separately during implementation

### 4. Integrity failure

Trigger:

1. Checksum mismatch
2. Artifact format mismatch
3. Missing required model artifacts in a native success payload
4. Missing tokenizer artifact for families that require one
5. `_ensure_safetensors(...)` fails for an Orbax-backed committed result

Behavior:

1. Raise `ValueError`
2. Treat the failure as terminal and non-retryable without upstream content change
3. Delete the committed final cache directory before raising if the failure is discovered after native commit
4. Do not leave a partially converted safetensors directory visible as a valid cache hit

Tokenizer rule:

1. If the model family requires a tokenizer and the committed result does not provide `tokenizer.model`, treat that as integrity failure, not as a best-effort omission
2. Families that intentionally do not require a tokenizer must be explicitly identified by the native contract rather than inferred from Python bucket layout

### 5. Invalid returned paths

Trigger:

1. Native downloader reports a cache directory outside `self.cache_path`
2. Path normalization escapes the cache root via `..`, symlink tricks, or malformed absolute paths
3. Required response fields are absent or inconsistent with the committed location

Behavior:

1. Raise `ValueError`
2. Reject the result before returning any path to callers
3. Delete the associated path if the native layer already materialized it
4. Never accept or normalize an out-of-root path into a caller-visible cache hit

## Cleanup Contract

The cutover is not acceptable unless cleanup obeys these rules:

1. The final cache directory becomes visible only after an atomic commit step
2. Staging directories are never considered valid cache hits
3. Any failure before commit removes staging state
4. Any integrity failure found after commit deletes the committed final directory
5. Existing valid warm-cache directories for prior successful downloads are never deleted because a new cold-path attempt fails

## Required Tests

The implementation PR should add or update tests that prove:

1. `NOT_FOUND` still raises `HubManager.ModelNotFoundError`
2. transport interruption raises `ConnectionError` and leaves no final cache tree
3. local disk failure raises `OSError` and is not remapped to `ConnectionError`
4. checksum or artifact mismatch raises `ValueError` and deletes the final cache tree
5. invalid returned paths raise `ValueError` and are rejected before returning to callers
6. a rerun after interrupted download is a clean cold-path retry, not a warm-cache false positive

## Follow-On Implementation Notes

1. [hub.py](/home/cody/code/c/mogemma/src/py/mogemma/hub.py) should own Python-side exception mapping and cache-root validation
2. The native downloader contract should report enough structured error detail to distinguish `NOT_FOUND`, transport, local I/O, integrity, and invalid-path failures
3. [model.py](/home/cody/code/c/mogemma/src/py/mogemma/model.py) should continue to see only committed cache directories or raised exceptions; it should not participate in download recovery
