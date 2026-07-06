# Learnings: Gemma 4 12B Unified Runtime

- Official 12B config fetched from Hugging Face on 2026-07-06 uses top-level `model_type = "gemma4_unified"` and nested `text_config.model_type = "gemma4_unified_text"`.
- 12B text geometry differs from the current runtime assumption: sliding attention uses `head_dim = 256`, while full/global attention uses `global_head_dim = 512` and `num_global_key_value_heads = 1`.
- 12B layer type names are `"sliding_attention"` and `"full_attention"`, not the existing local `"sliding"` / `"full"` fixture names.
- The public HF raw URL for `model.safetensors.index.json` returned `Entry not found` on 2026-07-06, so the next runtime slice must be local-safetensors-first and must not imply GCS/HF remote download support.

## [2026-07-06 03:25] - Phase 2 Task 2: CPU Variable-Head Runtime Geometry

- **Implemented:** CPU runtime geometry now stores per-layer head dimensions and KV strides for mixed sliding/full Gemma 4 layers; GPU requests with variable layer geometry are explicitly gated before GPU cache initialization.
- **Files changed:** `src/mo/mogemma/model.mojo`, `src/mo/mogemma/core.mojo`, `src/mo/mogemma/layers.mojo`, `src/mo/mogemma/gpu_context.mojo`, `src/py/tests/integration/test_mojo_core_contract.py`, Mojo unit fixtures.
- **Commit:** recorded in Beads task closure
- **Learnings:**
  - Patterns: Prefer tensor-projected dimensions (`q_proj.shape_0`, cache layer stride) over global `num_heads * head_dim` in attention code when layer geometry can vary.
  - Gotchas: GPU KV cache still mirrors a uniform stride layout; variable CPU geometry must raise before GPU resource initialization.

## [2026-07-06 04:05] - Phase 3 Task 3: 12B Text-Only Runtime Gate

- **Implemented:** Python runtime gating now permits Gemma 4 12B generation initialization on CPU, keeps 12B embeddings unsupported, keeps 12B GPU generation explicitly gated, and rejects 12B image/audio inputs before hydrator work.
- **Files changed:** `src/py/mogemma/model.py`, `src/py/mogemma/model_support.py`, `src/py/tests/unit/test_model_config.py`, `src/py/tests/unit/test_media.py`.
- **Commit:** recorded in Beads task closure
- **Learnings:**
  - Patterns: Store the detected Gemma 4 variant on `SyncGemmaModel` so stream-time modality gates do not need to re-read config files.
  - Gotchas: `_initialize_llm()` uses `_core.init_model_with_options()` when available, so CPU text-init tests should assert the core init path rather than only `backend.init_model()`.

## [2026-07-06 04:35] - Phase 4 Task 4: 12B Loader and Inventory Boundaries

- **Implemented:** Local Gemma 4 12B safetensors directories are accepted by `auto_loader()`, `google/gemma-4-12B-it` remains excluded from `KNOWN_GCS_MODELS` without live GCS proof, and unknown unified Orbax keys now raise before conversion is overclaimed.
- **Files changed:** `src/py/mogemma/convert.py`, `src/py/tests/unit/test_loader.py`, `src/py/tests/unit/test_hub.py`, `src/py/tests/integration/test_convert.py`.
- **Commit:** recorded in Beads task closure
- **Learnings:**
  - Patterns: Keep 12B text support local-safetensors-first until a real tensor inventory validates remote and Orbax layouts.
  - Gotchas: blocked: no local 12B checkpoint was available in `MOGEMMA_12B_LOCAL`, so no `12b-inventory.txt` artifact was generated.

## [2026-07-06 05:05] - Phase 5 Task 5: Docs and Final Verification

- **Implemented:** README and knowledge docs now state that Gemma 4 12B is official, local-safetensors CPU text runtime is supported, and unified image/audio, GPU, GCS download, and Orbax conversion remain follow-up gates.
- **Files changed:** `README.md`, `.agents/knowledge/gemma4-models.md`, `.agents/knowledge/python-runtime.md`, `.agents/flows.md`.
- **Commit:** recorded in Beads task closure
- **Verification:** `make build`, `make lint`, `uv run pytest -q src/py/tests`, and `uv run pytest -q src/mo/tests/test_mojo.py` passed.
- **Learnings:**
  - Patterns: Keep official status, GCS availability, and local runtime support as separate claims in docs.
  - Gotchas: `make lint` still emits existing Mojo `MutExternalOrigin` / `UnsafeAnyOrigin` deprecation warnings but exits 0; these warnings are not part of this 12B text support slice.
