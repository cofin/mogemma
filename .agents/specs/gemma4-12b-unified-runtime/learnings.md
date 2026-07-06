# Learnings: Gemma 4 12B Unified Runtime

- Official 12B config fetched from Hugging Face on 2026-07-06 uses top-level `model_type = "gemma4_unified"` and nested `text_config.model_type = "gemma4_unified_text"`.
- 12B text geometry differs from the current runtime assumption: sliding attention uses `head_dim = 256`, while full/global attention uses `global_head_dim = 512` and `num_global_key_value_heads = 1`.
- 12B layer type names are `"sliding_attention"` and `"full_attention"`, not the existing local `"sliding"` / `"full"` fixture names.
- The public HF raw URL for `model.safetensors.index.json` returned `Entry not found` on 2026-07-06, so the next runtime slice must be local-safetensors-first and must not imply GCS/HF remote download support.
