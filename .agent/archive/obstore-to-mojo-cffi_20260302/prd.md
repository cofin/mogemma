# Master PRD: Obstore to Mojo CFFI Migration

## Context

`mogemma` currently performs model discovery and download in Python via `obstore` (`src/py/mogemma/hub.py`). The inference boundary to Mojo is currently `_core` (`src/mo/mogemma/core.mojo`) with exported functions for `init_model`, `step`, and `generate_embeddings`, but no storage/download interface.

Goal from request:

- Replace obstore-based calls in `HubManager.download()` with a Rust-backed downloader exposed through Mojo CFFI.
- Use a hard cutover model with no backward compatibility fallback.
- Prioritize performance first, then reliability, then dependency reduction.
- Plan architecture/research thoroughly before implementation.

## North Star

`HubManager.download()` is powered by a Rust+Mojo CFFI download/data path (not Python obstore), with explicit performance and reliability acceptance gates and no dual-path compatibility layer.

## Roadmap

### Chapter 1: `obstore-cffi-architecture_20260302`

Define architecture and interface contracts for Rust↔Mojo↔Python integration, hard-cutover semantics, and measurable acceptance criteria.

### Chapter 2: `rust-downloader-ffi-contract_20260302`

Plan Rust downloader API and binary/FFI surface for use through Mojo CFFI, including ownership, error mapping, and transfer primitives.

### Chapter 3: `hub-hard-cutover_20260302`

Plan the Python hub integration and deletion scope to remove direct obstore usage from the active download path without fallback.

### Chapter 4: `perf-reliability-gates_20260302`

Plan and codify benchmark/reliability gates and release acceptance process for the migrated path.

## Global Constraints

- No backward compatibility path; this is a hard migration.
- Replacement scope targets `HubManager.download()` path first.
- Keep API contracts explicit and testable at each boundary.
- Migration must define deterministic failure behavior, not silent fallback.
- Performance and reliability gates are required before declaring migration complete.

