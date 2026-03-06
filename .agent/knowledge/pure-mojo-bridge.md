# Knowledge: Pure Mojo Bridge

> Flow: pure-mojo-bridge | Archived 2026-02-25

## Key Changes

Stripped all `modular` and `max` dependencies. Created `SafetensorsLoader` for memory-mapped weight loading. Implemented Python-to-Mojo zero-copy FFI layer using raw memory pointers.

## Patterns

- **Zero-copy bridging:** `tensor.data_ptr()` extracts memory addresses; Mojo accepts raw pointers via `UnsafePointer`.
- **Dictionary metadata passing:** Name -> [pointer, shape] mapping between Python and Mojo.
- **Dependency purge:** Removing heavy dependencies (`max`, `modular`) was prerequisite for all subsequent work.

## Gotchas

- Memory alignment must be verified when casting raw pointers.
- Multi-file safetensors require parsing `model.safetensors.index.json`.
