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

## [2026-05-06 21:33 UTC] - Phase 4 Verification Gates

- **Implemented:** closed local lint/type/test gates and deferred the live checkpoint manual gates that require uncached multi-GB model downloads.
- **Files changed:** `src/mo/mogemma/gpu_context.mojo`, `src/mo/tests/test_gpu_context.mojo`, `src/mo/tests/test_gpu_forward.mojo`, `.agents/specs/orbax-safetensors-conversion/learnings.md`
- **Verification:**
  - `uv run ruff check src/py/mogemma/convert.py src/py/mogemma/hub.py src/py/mogemma/orbax_loader.py` -> clean.
  - `uv run ruff format --check src/py/mogemma/` -> 13 files already formatted.
  - `uv run mypy src/py/mogemma/convert.py` -> no issues.
  - `CI=true uv run pytest src/py/tests/test_convert.py src/py/tests/test_orbax_loader.py src/py/tests/test_hub.py` -> 90 passed.
  - `CI=true uv run pytest src/py/tests/ -q` -> 295 passed, 4 skipped.
  - `uv run pytest src/mo/tests/test_mojo.py -q` -> 15 passed, 1 skipped after the Mojo pointer API fix.
  - `make test` -> 310 passed, 5 skipped.
  - `make lint` -> clean.
- **Learnings:**
  - **Gotcha:** Current Mojo returns an `UnsafePointer` directly from `HostBuffer.unsafe_ptr()`. Legacy `.value()` accessors no longer compile; use the returned pointer directly for arithmetic, `load`, and `store`.
  - **Context:** The E2B and 26B manual verification cache slices are absent on this host. `/home/cody` has 99G free, which is not enough to safely cold-download both E2B (~17G from the prior inventory note) and 26B (~52G from the MoE runtime spec) plus conversion/output headroom in one cleanup pass.
  - **Deferral:** Task 4.2 remains a live/manual checkpoint-gate exercise for a machine with sufficient cache space and bandwidth. The local automated conversion, loader, and Mojo harnesses are green.
