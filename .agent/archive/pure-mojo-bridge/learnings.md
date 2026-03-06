# Learnings: Pure Mojo Bridge

## [2026-02-25] - Dependency purge & zero-copy FFI

- **Implemented:** Removed `modular`/`max` dependencies. SafetensorsLoader with zero-copy memory bridging.
- **Learnings:**
  - `tensor.data_ptr()` extracts memory addresses; Mojo accepts via `UnsafePointer`.
  - Memory alignment must be verified when casting raw pointers.
  - Multi-file safetensors require parsing `model.safetensors.index.json`.
  - Dependency purge is prerequisite for all subsequent Pure Mojo work.
