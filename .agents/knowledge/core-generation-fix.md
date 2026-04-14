# Learnings: core-generation-fix

## 2026-02-24

- Gemma instruction-tuned checkpoints (`*-it`) require turn formatting to achieve reliable factual outputs. Raw prompts can produce low-quality multiple-choice artifacts even when model weights are healthy.
- A deterministic validation gate (`temperature=0`, `top_k=1`) is required for factual contract checks; stochastic decoding masks regressions.
- Nano (`gemma3n-e2b-it`) remains unresolved: output is still incoherent after fixing one AltUp correction path bug, indicating deeper architecture mismatch in Mojo nano path.
- Nano conversion/runtime must treat per-layer dimensions as authoritative from `per_layer_map.gate.weight` and validate AltUp/per-layer tensor layouts at conversion time; inferring `per_layer_dim` from `per_layer_embed.shape_1` is unsafe with 3D embeddings.
- For Gemma3n, conversion-shape correctness is necessary but insufficient; output quality depends on reference-faithful runtime semantics (AltUp + per-layer-input + KV-sharing behavior) and on eliminating hot-path model rebuild overhead by caching runtime descriptors at init.
- The factual Nano breakage was architectural, not bad weights: layers `20..29` in `gemma3n-e2b-it` have zeroed `k_proj`/`v_proj`, so KV-sharing must be implemented; treating those layers as normal attention causes degenerate generation.
- Gemma3n parity requires Nano-specific norms/attention behavior: weighted RMSNorm (without `1 + w`) for nano norm tensors, `v_norm` on values, attention scaling of `1.0`, and correct AltUp `prediction_coefs` axis order.
- Gemma3n early decoder layers depend on activation sparsity in MLP gates (Gaussian top-k style, default `0.95` for first 10 layers); skipping this step materially degrades output quality.
- Instruction-turn termination should prioritize `<end_of_turn>` as EOS for `*-it` models, otherwise generation drifts into repeated turn markers.
- Caching parsed Mojo model descriptors behind per-session slots (`model_slot`) removes repeated `_build_*_model_from_runtime` work from `step_mojo` and embedding paths; the hot path should only read cached descriptors.
- Descriptor build counters used for regression tests should tolerate older prebuilt cores that do not expose the counter key yet; tests can default to an expected steady-state count of `1`.
