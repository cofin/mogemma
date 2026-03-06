# Flow: rust-downloader-ffi-contract_20260302

## Specification

### Goal

Define implementation-ready contract for Rust downloader APIs exposed through Mojo CFFI, including ownership, streaming/buffer model, and error mapping.

### Decision Snapshot (2026-03-02)

1. `Choice 1 = A`: Rust downloader owns write-to-cache behavior and exposes it through Mojo CFFI entrypoints.
2. `Choice 2 = B`: use explicit retry/error taxonomy at the contract boundary.
3. `Choice 3 = C`: move tokenizer selection into Mojo boundary contract.
4. `Choice 4 = B`: prioritize high-throughput concurrent transfer model.
5. `Choice 5 = C`: `_ensure_safetensors` integration deferred under follow-up issue `mogemma-h4n.5`.

### Requirements

1. Function-level API contract for listing and downloading model artifacts.
2. Defined ownership/lifetime rules for buffers and file handles across FFI boundary.
3. Error code taxonomy mapped to Python exceptions and user-facing messages.
4. Concurrency and cancellation semantics consistent with hard-cutover expectations.

## Implementation Plan

### Phase 1: Rust FFI surface

- [x] 1.1 `mogemma-h4n.2.1` Define Rust write-to-cache downloader API (request/response structs, cache root, artifact selectors).
- [x] 1.2 `mogemma-h4n.2.2` Define retry/error taxonomy and retryability metadata for deterministic propagation.

### Phase 2: Mojo interop boundary

- [x] 2.1 `mogemma-h4n.2.5` Define tokenizer/artifact selection contract inside Mojo-facing layer.
- [x] 2.2 `mogemma-h4n.2.4` Define concurrency model (parallel segment/object fetch, cancellation, backpressure).

### Phase 3: Integration contract output

- [x] 3.1 `mogemma-h4n.2.3` Produce file-level implementation contract consumed by Chapter 3 cutover tasks.
- [ ] 3.2 Define interface contract validation strategy (ABI checks, error mapping checks, deterministic failure assertions).

## Task Acceptance Criteria

### `mogemma-h4n.2.1` Rust write-to-cache downloader API

1. API contract defines required inputs: model id, revision, cache root, artifact selection mode.
2. API contract defines outputs: resolved local paths, artifact manifest, byte totals.
3. Contract explicitly states atomic file-write behavior and cache consistency guarantees.

### `mogemma-h4n.2.2` Retry/error taxonomy

1. Error classes are enumerated and stable (network transient, auth, not found, integrity, internal).
2. Each error class includes retryability and backoff policy guidance.
3. Error-to-Python exception mapping is deterministic and documented.

### `mogemma-h4n.2.5` Tokenizer selection in Mojo layer

1. Tokenizer selection source-of-truth is defined at Mojo boundary (not Python hub logic).
2. Contract covers tokenizer artifact discovery, normalization, and returned local path semantics.
3. Contract specifies behavior when tokenizer artifacts are absent or mismatched.

### `mogemma-h4n.2.4` High-throughput concurrency model

1. Concurrency model defines worker/queue behavior and max parallelism controls.
2. Cancellation and timeout semantics are explicit across Rust, Mojo, and Python boundary.
3. Model includes backpressure/limit behavior to avoid unbounded memory growth.

### `mogemma-h4n.2.3` Chapter 3 implementation contract

1. Contract names exact Chapter 3 integration touchpoints in `src/py/mogemma/hub.py`.
2. Contract includes sequence diagram or equivalent ordered call flow from hub -> Mojo -> Rust.
3. Contract includes a checklist of invariants Chapter 3 must preserve (cache layout, strict errors, no fallback).

## Task Artifacts

1. `mogemma-h4n.2.1`: `.agent/specs/rust-downloader-ffi-contract_20260302/rust-write-to-cache-api-contract.md`
2. `mogemma-h4n.2.2`: `.agent/specs/rust-downloader-ffi-contract_20260302/retry-error-contract.md`
3. `mogemma-h4n.2.4`: `.agent/specs/rust-downloader-ffi-contract_20260302/high-throughput-concurrency-contract.md`
4. `mogemma-h4n.2.5`: `.agent/specs/rust-downloader-ffi-contract_20260302/tokenizer-selection-mojo-boundary-contract.md`
5. `mogemma-h4n.2.3`: `.agent/specs/rust-downloader-ffi-contract_20260302/chapter3-implementation-contract.md`
