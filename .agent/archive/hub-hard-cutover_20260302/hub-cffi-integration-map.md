# Hub CFFI Integration Map (`mogemma-h4n.3.1`)

## Objective

Map every active `obstore`-based operation in `HubManager.download()` to the replacement Mojo CFFI downloader path.

## Current Active Pattern (to replace)

1. Build remote store client (GCS-backed).
2. List remote artifact objects for a model/revision.
3. Iterate remote objects and fetch each object body.
4. Materialize objects into local cache paths.
5. Return resolved local model directory and tokenizer path behavior to callers.

## Target Hard-Cutover Pattern

1. Normalize model request in Python hub layer (`model_id`, `revision`, cache location).
2. Perform a single Mojo CFFI downloader invocation.
3. Mojo boundary applies tokenizer/artifact selection policy.
4. Rust downloader resolves and materializes artifacts atomically to cache.
5. Python receives deterministic result payload and returns resolved local paths.

## Integration Point Mapping

## Mapping A: Remote listing

1. Replace Python-side remote list operation with Mojo call:
1. `cffi_download_to_cache(request)` handles discovery and transfer together.
2. Pre-condition: normalized `model_id`, optional `revision`.
3. Post-condition: response contains committed artifact set or deterministic terminal error.

## Mapping B: Per-artifact fetch loop

1. Remove Python artifact iteration/fetch loop from active path.
2. Rust downloader owns artifact-level scheduling and retries.
3. Pre-condition: bounded concurrency value in request.
4. Post-condition: no partially committed artifacts are treated as valid cache outputs.

## Mapping C: Cache materialization

1. Remove Python-managed write loop for fetched object bytes.
2. Rust downloader performs staging + atomic commit into cache root.
3. Pre-condition: cache root path is absolute and writable.
4. Post-condition: returned paths point to committed artifacts only.

## Mapping D: Tokenizer resolution

1. Remove Python-local tokenizer selection logic from active path.
2. Mojo boundary returns tokenizer resolution outcome.
3. Pre-condition: selection policy provided in request.
4. Post-condition: `tokenizer_path` resolved or deterministic `NOT_FOUND`/`INTEGRITY` failure.

## Mapping E: Error surfacing

1. Replace Python ad hoc download errors with boundary category mapping.
2. Mojo forwards stable categories from Rust (`NOT_FOUND`, `AUTH`, `NETWORK_*`, `INTEGRITY`, `TIMEOUT`, `CANCELLED`, `INTERNAL`).
3. Python raises deterministic mapped exceptions with model context.

## Invariants for Implementation

1. No runtime fallback to `obstore` list/get in the active download path.
2. `HubManager.download()` remains the single entry for caller-visible download behavior.
3. Cache layout semantics required by downstream model loading remain stable.
4. Failure semantics are explicit and deterministic.
