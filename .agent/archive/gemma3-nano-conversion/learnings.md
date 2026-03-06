# Learnings: Gemma3 Nano Conversion

## [2026-02-25] - Orbax-to-safetensors conversion pipeline

- **Implemented:** `_convert_gemma3_nano()` for 671-tensor layout. Model family auto-detection. Shared conversion helpers.
- **Learnings:**
  - Nano has 671 tensors vs standard's 237-341 — fundamentally different architecture.
  - Model family detection via presence of `altup`/`per_layer_mapping` tensor keys.
  - 3D per-layer embeddings (262144x30x256) handled as single contiguous block.
  - Attention uses `query_norm` (not `_query_norm` like standard) — naming inconsistency.
  - Shared conversion logic (attention, MLP, norms) extracted to DRY helpers.
