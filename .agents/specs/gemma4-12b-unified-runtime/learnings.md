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
