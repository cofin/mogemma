# Flow: obstore-cffi-architecture_20260302

## Specification

### Code Analysis Summary

Files analyzed:

- `src/py/mogemma/hub.py`
- `src/py/mogemma/model.py`
- `src/py/tests/test_hub.py`
- `src/mo/mogemma/core.mojo`
- `pyproject.toml`

Key findings:

1. Active obstore calls are centralized in `HubManager.download()` (`obs.store.GCSStore`, `obs.list`, `obs.get`) in `src/py/mogemma/hub.py`.
2. `resolve_model(..., download_if_missing=True, strict=True)` in `src/py/mogemma/model.py` routes model resolution into `HubManager.download()`; this is the primary integration trigger.
3. `_core` Mojo FFI (`src/mo/mogemma/core.mojo`) currently exports inference functions only (`init_model`, `step`, `generate_embeddings`) and has no download/storage APIs.
4. The current Python downloader writes directly into the final cache directory with a `ThreadPoolExecutor(max_workers=8)`, so interrupted downloads can leave partially materialized cache trees that look locally valid.
5. Existing hub tests in `src/py/tests/test_hub.py` validate local/cached/strict behavior, but they do not yet lock atomic commit, partial-download cleanup, or stable downloader error categories.
6. Obstore is declared as a runtime dependency in `pyproject.toml`; hard cutover implies dependency and call-path changes.

### Decision Contract for This Flow

The following decisions are now assumed for downstream implementation unless a later revision explicitly changes them:

1. Rust owns remote discovery, bounded concurrency, retries, checksums/integrity checks, staging, and final atomic cache commit.
2. Python performs one downloader invocation per `HubManager.download()` call and never iterates remote artifact bodies in the hot path.
3. The Python-visible cache directory name remains `model_id.replace("/", "--")` so existing callers and tests keep the same local resolution semantics.
4. Tokenizer/artifact selection policy is owned by the Rust+Mojo boundary, and Python consumes a resolved `tokenizer_path` from the response instead of reproducing bucket-layout rules.
5. `_ensure_safetensors()` remains Python-orchestrated in the initial cutover, but it may run only after the downloader reports a fully committed cache tree and the response explicitly flags Orbax content.

### Architecture Contract

#### Python request payload

`HubManager.download()` must be able to construct a request object with these fields, even if the exact struct name changes in Chapter 2:

1. `model_id`: original caller-provided identifier.
2. `clean_model_id`: normalized bucket-facing identifier after `_clean_model_id`.
3. `cache_root`: absolute cache root path supplied by `HubManager`.
4. `target_dir_name`: stable final directory name derived from `model_id`.
5. `revision`: optional revision string, defaulting to upstream default branch when absent.
6. `max_concurrency`: bounded downloader parallelism hint, default 8 and capped at 16 unless benchmarks prove higher is better.
7. `allow_resume`: enabled for retriable partial staging reuse, disabled for final-path mutation.

#### Python response payload

The downloader boundary must return enough data to let Python validate correctness without reopening the artifact stream:

1. `cache_dir`: committed final directory path under `cache_root`.
2. `tokenizer_path`: resolved local tokenizer path or explicit null when the model family does not require one.
3. `artifact_manifest`: deterministic list of relative artifact paths, sizes, and integrity metadata.
4. `downloaded_bytes` and `elapsed_ms`: for perf evidence and log correlation.
5. `retry_count`: total retry attempts consumed by the Rust path.
6. `used_revision`: concrete upstream revision/commit used for the cache result.
7. `contains_orbax`: boolean used to decide whether `_ensure_safetensors()` should run.

#### Error contract

All cross-boundary failures must map to stable categories before they reach Python:

1. `NOT_FOUND`: model or required tokenizer/artifact does not exist upstream.
2. `NETWORK`: transport timeout/reset/DNS/TLS style failures after retry exhaustion.
3. `AUTH`: bucket access/auth policy failures.
4. `INTEGRITY`: checksum mismatch, short read, corrupt archive, or invalid manifest.
5. `LOCAL_IO`: staging directory creation, disk-full, rename, or permissions failures.
6. `CANCELLED`: caller-initiated cancellation or process termination semantics surfaced intentionally.
7. `INTERNAL`: downloader invariant violations that should fail closed.

### Requirements

1. Define target architecture and function-level contracts for Rust downloader exposure through Mojo CFFI.
2. Define explicit memory ownership, staging/commit lifecycle, and error propagation semantics across Rust↔Mojo↔Python boundaries.
3. Define hard-cutover semantics for removing the obstore path from `HubManager.download()` with no runtime fallback.
4. Define performance and reliability design constraints that downstream implementation must satisfy before Chapter 3/4 completion.

### Performance-Critical Constraints

1. The Python hot path must remain `O(1)` in FFI calls per download request; it may not perform per-artifact body reads, per-chunk callbacks, or Python-thread scheduling for transfer work.
2. Final cache directories must be written through a staging path and promoted atomically so downstream model resolution never observes a partially committed cache tree.
3. Manifest validation must be derived from downloader output; Python may perform final path/existence checks, but it must not rescan every artifact just to reconstruct downloader state.
4. Concurrency must be bounded and explicit. The initial contract should target a default of 8 workers and a hard cap of 16 until benchmark evidence in Chapter 4 proves a different default is materially better.
5. Python-side memory growth during download should be bounded by request/response metadata; artifact bytes are not allowed to transit through Python objects on the steady-state path.

### Interop Compatibility Guardrails

1. The downloader boundary must validate against the current stable Mojo line (`v0.26.1`) before any nightly-only workaround is accepted into the default path.
2. Python interop coverage for this flow must include the currently documented supported CPython range (`3.10`-`3.14`) when the boundary is exercised in CI or release qualification.
3. If the implementation depends on Mojo/Python buffer-protocol helpers or other newer interop APIs, the spec must name the exact minimum Mojo version required and keep older behavior out of the default path.

### Required Validation Evidence

1. A call graph artifact showing every current `hub.py` obstore touchpoint and its replacement boundary call.
2. A request/response contract table covering request fields, response fields, error categories, and ownership of each field.
3. A lifecycle diagram for `staging -> integrity verification -> atomic commit -> optional safetensors conversion`.
4. A preservation matrix tying current `test_hub.py`, `test_gemma_model.py`, and `test_embeddings.py` behaviors to either preserved or intentionally revised semantics.

## Implementation Plan

### Phase 1: Baseline and call graph

- [ ] 1.1 Document the exact current obstore download call graph in `src/py/mogemma/hub.py`, including where cache directories are created, where listing happens, and where file bodies are materialized (`mogemma-h4n.1.1`)
- [ ] 1.2 Enumerate caller-visible behavior contracts to preserve or intentionally revise from `src/py/tests/test_hub.py`, `src/py/tests/test_gemma_model.py`, and `src/py/tests/test_embeddings.py` (`mogemma-h4n.1.1`)

Phase 1 output must make it impossible to miss any active obstore dependency or any observable cache-resolution behavior expected by model initialization.

### Phase 2: Architecture contract

- [ ] 2.1 Define the Rust↔Mojo↔Python API contract, including exact request/response fields, path invariants, and ownership of staging/final cache directories (`mogemma-h4n.1.2`)
- [ ] 2.2 Define the structured error model with retryability rules, stable error categories, and caller-facing exception mapping (`mogemma-h4n.1.2`)

Phase 2 must explicitly state which layer owns tokenizer selection, which layer owns integrity checking, and which fields are required to produce benchmark evidence without re-reading artifacts in Python.

### Phase 3: Hard cutover and gates

- [ ] 3.1 Define hard-cutover deletion scope plus failure semantics for missing models, partial staging state, integrity failures, and local disk errors (`mogemma-h4n.1.3`)
- [ ] 3.2 Define performance/reliability acceptance targets and the exact evidence package required from the implementation PR (`mogemma-h4n.1.4`)
- [ ] 3.3 Checkpoint: produce an implementation-ready Chapter 2 execution contract that names target files, required interface deltas, and blocking validation evidence

Phase 3 must leave the implementation PR with no unresolved questions about whether Python still owns artifact iteration, how partial downloads are cleaned up, or what benchmark/reliability evidence blocks release.

### Phase 4: Compatibility and contract verification

- [ ] 4.1 Define the Mojo/Python support matrix used to validate the downloader boundary (stable channel first, nightly only if explicitly justified).
- [ ] 4.2 Define contract tests for ownership/freeing, path validation, and deterministic error mapping at the FFI seam.
- [ ] 4.3 Checkpoint: no implementation work starts until ABI ownership, retry policy, and cache-atomicity guarantees are all testable.

## TDD-Oriented Task Checklist

- [ ] RED: add failing contract tests for response ownership/freeing and stable error-category mapping.
- [ ] RED: add failing tests proving interrupted or integrity-failed downloads leave no committed cache tree.
- [ ] GREEN: implement the boundary contract and cleanup semantics until the red tests pass.
- [ ] REFACTOR: centralize request/response shaping and path-validation helpers without changing caller-visible semantics.
- [ ] REGRESSION: rerun `src/py/tests/test_hub.py` plus new boundary tests across the supported Python matrix.
