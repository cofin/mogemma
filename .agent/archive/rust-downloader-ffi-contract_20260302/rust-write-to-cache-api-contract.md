# Rust Write-to-Cache API Contract (`mogemma-h4n.2.1`)

## Purpose

Define the Rust downloader interface used through Mojo CFFI for deterministic artifact materialization into local cache.

## API Surface (Contract Level)

## `download_to_cache(request) -> response`

Required request fields:
1. `model_id: str` normalized model identifier.
2. `revision: str | None` optional revision pointer.
3. `cache_root: str` absolute cache root path.
4. `artifact_selection: str` one of `model_only` or `model_plus_tokenizer`.
5. `concurrency: int` bounded parallelism hint.
6. `timeout_ms: int` total request budget.

Response fields:
1. `resolved_model_dir: str` canonical local directory for model artifacts.
2. `artifact_paths: list[str]` full local paths for downloaded artifacts.
3. `tokenizer_path: str | None` resolved tokenizer path when requested.
4. `bytes_downloaded: int` total transferred bytes.
5. `cache_hit: bool` indicates all requested artifacts were already present.
6. `diagnostics: str | None` optional non-fatal operational metadata.

## `list_remote_artifacts(request) -> response`

Required request fields:
1. `model_id: str`
2. `revision: str | None`
3. `artifact_selection: str`

Response fields:
1. `artifacts: list[ArtifactRecord]`
2. `remote_manifest_hash: str | None`

`ArtifactRecord` contract fields:
1. `name: str`
2. `size_bytes: int`
3. `checksum: str | None`
4. `kind: str` (`model` or `tokenizer`)

## Cache Write Semantics

1. Rust downloader owns all file writes for active path.
2. Writes use staging files and atomic commit/rename.
3. Failed or cancelled transfers must not appear as committed cache artifacts.
4. Cache path layout is deterministic from `model_id`, `revision`, and artifact name.

## Ownership and Lifecycle

1. Rust owns transfer buffers and file handles until response completion.
2. Mojo boundary receives immutable response payload and forwards to Python.
3. Python never mutates downloader-managed temporary state.

## Deterministic Failure Contract

Error categories:
1. `NOT_FOUND`
2. `AUTH`
3. `NETWORK_TRANSIENT`
4. `NETWORK_PERMANENT`
5. `INTEGRITY`
6. `INTERNAL`

Each error payload includes:
1. `category`
2. `message`
3. `retryable: bool`
4. `artifact_name: str | None`

## Concurrency Boundaries

1. `concurrency` is capped by implementation-defined max.
2. Internal worker pool must enforce bounded queue/backpressure.
3. Cancellation interrupts active transfers and returns deterministic cancellation error.

## Chapter 3 Handoff Notes

1. `HubManager.download()` should call Mojo boundary once per model request.
2. Python-side `obstore` list/get loop is removed in hard cutover path.
3. Returned `resolved_model_dir` is the only source of truth for downstream model loading.
