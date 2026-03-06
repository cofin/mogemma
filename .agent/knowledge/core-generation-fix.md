# Knowledge: Core Generation Fix

> Flow: core-generation-fix | Active (2026-02-24)

## Gemma IT Model Patterns

- Instruction-tuned (`*-it`) checkpoints require turn formatting for reliable factual output. Raw prompts produce low-quality artifacts.
- Deterministic validation gate (`temperature=0`, `top_k=1`) required for factual contract checks.
- Prioritize `<end_of_turn>` as EOS for `*-it` models to prevent generation drift.

## Gemma3 Nano Architecture

- **KV Sharing:** Layers 20..29 in `gemma3n-e2b-it` have zeroed `k_proj`/`v_proj` — must implement KV-sharing, not normal attention.
- **Weighted RMSNorm:** Nano norms use `w * x` (not `1 + w`), plus `v_norm` on values, attention scaling of `1.0`.
- **Activation Sparsity:** Early decoder layers (first 10) need Gaussian top-k gating (default `0.95`).
- **AltUp Dimensions:** Per-layer dims authoritative from `per_layer_map.gate.weight`; inferring from `per_layer_embed.shape_1` unsafe with 3D embeddings.
- **Conversion vs Runtime:** Shape-correct conversion is necessary but insufficient — runtime must faithfully implement AltUp + per-layer-input + KV-sharing.
- Cache runtime descriptors at init to avoid hot-path model rebuild overhead.
