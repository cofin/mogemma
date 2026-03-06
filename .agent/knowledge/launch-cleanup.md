# Knowledge: Launch Cleanup

> Flow: launch-cleanup | Archived 2026-02-22

## Key Changes

Removed all vision-related code (files, exports, tests, Mojo functions). Removed broken test files, unused decorators, dummy fallbacks. Renamed `GemmaModel` -> `SyncGemmaModel`. Removed `mojo` from runtime deps (build-only).

## Patterns

- **Dead code identification:** Unused exports, broken imports, stub implementations.
- **Fail-fast over fallbacks:** Replaced dummy generation fallback with `RuntimeError`.
- **Lazy imports:** `__init__.py` lazy imports make selective removal cleaner.
- **Full codebase search** needed for cross-file references before any deletion.

## Gotchas

- Multiple broken Mojo test files existed importing deleted modules (`quant`, `inference`, `cache`, `vision`).
- `EmbeddingConfig.__post_init__` silently passed on invalid paths — config validation must be explicit.
- Mojo should never be a runtime dependency (build-only).
- `scripts/benchmark.py` was referenced but didn't exist (phantom reference).
