# Learnings: orbax-safetensors-conversion

## [2026-04-14] - Phase 0 Task 0.1: E2B-it inventory dump

- **Implemented:** scripts/dump_orbax_inventory.py + live GCS download of google/gemma-4-e2b-it
- **Files changed:** scripts/dump_orbax_inventory.py (ephemeral)
- **Learnings:**
  - **Gotcha:** GCS does NOT ship `config.json` for E2B-it (top-level is `_CHECKPOINT_METADATA`, `_METADATA`, `commit_success.txt`, `d`, `descriptor`, `manifest.ocdbt`, `ocdbt.process_0`, `tokenizer.model`). Conversion MUST generate config.json.
  - **Gotcha:** E2B-it checkpoint is 17GB (not ~5GB as initially assumed) — disk planning for multi-variant workflows must budget accordingly.
  - **Pattern:** `OrbaxLoader.__new__(cls)` stub pattern (populating `.model_path` only) exposes `_enumerate_tensor_names` / `_open_tensor` without triggering eager load. Used in dump script and new streaming helpers.

## [2026-04-14] - Phase 0 PLE mapping verification (layers.mojo:884-908)

- **Implemented:** resolved PLE Orbax→safetensors mapping ambiguity by reading `forward_ple_input` in layers.mojo.
- **Learnings:**
  - **Pattern:** when a contract is ambiguous from load-time code alone, read the forward-pass consumer. Tensor *shapes* used by `vec_mat_mul(out, vec, mat, M, N)` and `rms_norm(..., N, eps)` pin the expected `(M, N)` layout definitively.
  - Orbax carries 4 PLE-adjacent tensors the Mojo forward doesn't consume (`input_gate`, `skip_scale`, `model_projection`, `projection_norm`). Skip them — emitting them would invent safetensors names for nothing.
  - The `(1536,)` per-layer PLE norm is `layer_N.post_per_layer_input_norm.scale`, not the global `embedder.per_layer_projection_norm.scale` (which is `(256,)` — wrong axis).
