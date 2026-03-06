# Chapter 3 Implementation Contract (`mogemma-h4n.2.3`)

## Purpose

This contract defines the implementation handoff from Chapter 2 planning into Chapter 3 hard cutover work. It is the authoritative boundary for replacing `obstore`-driven download logic in `HubManager.download()` with a Mojo CFFI path backed by Rust downloader logic.

## Scope

1. Replace active download/list calls in `src/py/mogemma/hub.py` with Mojo CFFI invocations.
2. Keep hard cutover semantics: no runtime fallback to `obstore` for the active download flow.
3. Preserve cache-path and resolved-artifact behavior expected by model init and call sites.

## Non-Goals

1. No compatibility shim that keeps both downloader paths active.
2. No deferred `_ensure_safetensors` integration in this cutover (tracked by `mogemma-h4n.5`).
3. No broad refactor outside hub download and immediate boundary modules.

## File-Level Integration Contract

1. `src/py/mogemma/hub.py`
- `HubManager.download()` must call a single Mojo-facing downloader entrypoint for model artifacts.
- Python-side artifact iteration logic currently tied to `obstore list/get` must be removed from the active path.
- Returned local artifact paths must be concrete filesystem paths under the configured cache root.

2. `src/mo/mogemma/core.mojo` (or designated Mojo FFI boundary module if split)
- Expose downloader-facing CFFI entrypoint(s) callable from Python extension boundary.
- Implement tokenizer selection logic at this boundary, not in Python hub code.
- Return normalized status + payload suitable for deterministic Python exception mapping.

3. Rust downloader boundary module (new/updated implementation module)
- Own transfer execution, file materialization, retry handling, and integrity signaling.
- Write artifacts atomically to cache (tmp + commit/rename semantics).
- Emit stable error categories and retryability metadata.

## Data Contract (Boundary-Level)

## Request fields

1. `model_id`: normalized model identifier.
2. `revision`: optional revision/tag/commit pointer.
3. `cache_root`: absolute path where artifacts are materialized.
4. `artifact_mode`: model-only or model+tokenizer policy.
5. `concurrency`: explicit parallelism hint.
6. `timeout_ms`: overall timeout budget.

## Response fields

1. `resolved_model_dir`: canonical local model directory path.
2. `artifact_paths`: normalized artifact path list written to local cache.
3. `tokenizer_path`: optional resolved tokenizer path (if requested/available).
4. `bytes_downloaded`: total bytes transferred.
5. `cache_hit`: boolean summary for warm-path behavior.
6. `diagnostics`: optional non-fatal transfer metadata.

## Error contract

1. Stable categories: `NOT_FOUND`, `AUTH`, `NETWORK_TRANSIENT`, `NETWORK_PERMANENT`, `INTEGRITY`, `INTERNAL`.
2. Each error includes retryability metadata.
3. Python layer maps category -> deterministic exception type/message.
4. Hard failures abort the operation; no implicit fallback downloader.

## Concurrency and Cancellation Contract

1. Concurrency is bounded by explicit limit; unbounded worker growth is disallowed.
2. Cancellation from Python propagates to Mojo and Rust layers quickly and safely.
3. Partial artifacts from cancelled/failed runs are not left as valid cache entries.

## Ordered Call Flow (Chapter 3 target behavior)

1. Python `HubManager.download()` normalizes model request.
2. Python invokes Mojo CFFI downloader API once per model request.
3. Mojo boundary applies tokenizer/artifact selection policy.
4. Mojo invokes Rust downloader with normalized request.
5. Rust materializes artifacts into cache atomically and returns structured result/error.
6. Mojo maps result into boundary payload and returns to Python.
7. Python returns resolved local path(s) or raises deterministic mapped exception.

## Invariants Chapter 3 Must Preserve

1. Download path is deterministic and fallback-free.
2. Cache outputs are usable by existing model loading flow without additional mutation.
3. Failure behavior is explicit and user-visible.
4. Tokenizer resolution logic source-of-truth is Mojo boundary.
5. Active runtime no longer depends on `obstore` list/get in `HubManager.download()`.

## Chapter 3 Deliverable Checklist

1. `mogemma-h4n.3.1`: integration-point map reflects this contract exactly.
2. `mogemma-h4n.3.2`: failure semantics align with the error contract categories.
3. `mogemma-h4n.3.3`: removal scope proves no active `obstore` fallback path remains.
4. `mogemma-h4n.3.4`: tokenizer handoff is implemented per Mojo-boundary ownership rule.

## Validation Hooks (Planning-Level)

1. Contract test covering successful model+tokenizer download path and returned local paths.
2. Contract tests for each stable error category with deterministic Python exception mapping.
3. Contract test proving cancellation does not leave committed partial artifacts.
4. Contract check that active hub path does not call `obstore` list/get during download.
