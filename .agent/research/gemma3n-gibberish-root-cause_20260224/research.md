# Research: Gemma 3n gibberish output root cause (mogemma)

## Executive Summary

`gemma3n-e2b-it` gibberish generation is not explained by "bad weights". The converted tensor shapes look structurally correct, but the Mojo Nano forward pass is not architecture-faithful to published/production Gemma 3n behavior.

The highest-confidence factual issue is **forward semantics mismatch**, especially around AltUp and per-layer-input processing. In short, the code loads many Nano tensors but does not execute them with the same math/state transitions used by upstream implementations.

## Codebase Analysis

### 1) AltUp math mismatch (critical)

Local implementation:
- `src/mo/mogemma/layers.mojo:421` performs router + predict + unembed + scalar correction in one custom path.
- Router output is softmaxed (`src/mo/mogemma/layers.mojo:447`).
- Prediction uses fixed flattened indexing (`in_m * 16 + out_m * 4 + router_m`) (`src/mo/mogemma/layers.mojo:464`).
- Correction collapses to a single scalar `corr_combined` applied to all hidden dims (`src/mo/mogemma/layers.mojo:486-495`).

Upstream (Transformers Gemma3n):
- `Gemma3nTextAltUp.predict` computes router modalities with `tanh(router_norm(...))`, not softmax, then computes a matrix projection and adds original hidden states.
- `Gemma3nTextAltUp.correct` computes innovation vs active branch, applies per-branch correction coefficients `+1.0`, then adds predictions.
- Reference: `transformers/models/gemma3n/modeling_gemma3n.py:1098-1160` in installed package.

Implication:
- Current Mojo AltUp uses materially different probability/mixture semantics and correction behavior, which can destabilize logits into gibberish even with correct weights.

### 2) Per-layer input pipeline is incomplete (critical)

Local implementation:
- `model.per_layer_embed.projection.weight` and `model.per_layer_embed.norm.weight` are loaded in core (`src/mo/mogemma/core.mojo:117-119`) but not used in Nano forward.
- Per-layer mapping path directly uses lookup from `model.per_layer_embed.weight` and a gate/projection (`src/mo/mogemma/layers.mojo:320-356`).
- Gate path has no activation between gate matmul and elementwise multiply.

Upstream:
- `project_per_layer_inputs` projects base token embeddings into per-layer inputs, scales, reshapes, norms, and combines with per-layer token embeddings.
- Then each decoder layer applies `per_layer_input_gate` + activation + multiply with `per_layer_input` + projection + norm.
- Reference: `transformers/models/gemma3n/modeling_gemma3n.py:1620-1627`, `1668-1689`, `1755-1787`, `1434-1442`.

Implication:
- Major part of Gemma3n layer-conditioning signal is missing/altered in Mojo.

### 3) Hidden-state topology mismatch (high)

Local:
- Nano layer path operates effectively as one hidden stream (`forward_nano_layer` / `forward_nano_step`) and synthesizes AltUp effects in-place.

Upstream:
- Model carries `altup_num_inputs=4` hidden streams through each layer (`[num_altup_inputs, batch, seq, hidden]`), with predict/correct over these streams.
- Reference: `transformers/models/gemma3n/modeling_gemma3n.py:1704-1748`, `1370-1444`.

Implication:
- Stream-level behavior expected by AltUp is not fully represented.

### 4) Attention behavior mismatch: KV sharing and v_norm (high)

Local:
- Attention computes K/V every layer and stores per-layer caches (`src/mo/mogemma/layers.mojo:57-62`, nano step cache indexing `623-625`).
- No value normalization path (only q_norm/k_norm).

Upstream:
- Gemma3n supports KV-shared layers (`num_kv_shared_layers`) and `v_norm`.
- Reference: `transformers/models/gemma3n/modeling_gemma3n.py:1284-1345` and `1282`.

Implication:
- Even if dimensions match, recurrent/shared-cache semantics diverge from reference model.

### 5) Hardcoded per-layer embedding layer count (medium)

Local:
- Per-layer embedding lookup assumes 30 layers in offset math (`token_id * 30 * per_layer_dim + ...`) (`src/mo/mogemma/layers.mojo:343`).

Current e2b config is 30 layers, so this works today, but it is brittle.

## External Documentation and Prior Art

### Official model architecture references

- Hugging Face Gemma3n configuration defaults and parameters:
  - `altup_num_inputs=4`, `num_kv_shared_layers`, `hidden_size_per_layer_input`.
  - Source: `configuration_gemma3n.py` in transformers and model config for `google/gemma-3n-E2B-it`.
- Hugging Face Gemma3n model implementation:
  - AltUp predict/correct definitions and decoder-layer integration.
  - Per-layer input projection/normalization path.
- Gemma 3n model card:
  - Explicitly describes MatFormer nesting and recurrent parameter/cache strategy.

### Repository prior art

- Recent fix added tensor-layout validation and per-layer-dim derivation from gate tensor (`commit 6405450`), which removed one class of conversion mistakes but did not resolve gibberish.
- Existing task note already points to upstream semantic divergence as likely root cause (`bd show mogemma-yj1.2.1`).

## Evidence Collected

From local converted weights (`~/.cache/mogemma/gemma3n-e2b-it/model.safetensors`):
- Layer count: 30 (indices 0..29).
- Core shapes look plausible:
  - `model.embed_tokens.weight`: `(262144, 2048)`
  - `model.per_layer_embed.weight`: `(262144, 30, 256)`
  - `model.layers.0.altup.router.weight`: `(4, 2048)`
  - `model.layers.0.altup.prediction_coefs`: `(4, 4, 4)`
  - `model.layers.0.altup.correction_coefs`: `(4, 4)`

This supports that the failure is likely in forward semantics, not raw tensor corruption.

## Risk Assessment

1. **False-fix risk (high):** Isolated tweaks (single transpose/index fix) may not restore coherence because multiple coupled semantic mismatches exist.
2. **Regression risk (medium):** Refactoring Nano path can break current standard Gemma path if abstractions are mixed.
3. **Validation gap (high):** Without intermediate parity checks vs reference outputs, debugging by final text quality is slow and noisy.

## Recommended Approach

1. **Implement architecture-faithful Nano parity layer-by-layer**
- Rework AltUp to follow reference predict/correct semantics (router modality computation, coefficient projection, innovation correction, optional scaling).
- Carry 4 hidden streams explicitly through Nano layer flow.

2. **Implement full per-layer-input pathway**
- Use `model.per_layer_embed.projection.weight` + norm + scaling and combine with token per-layer inputs before decoder.
- In each layer apply gate activation before multiply and post-projection norm as in reference.

3. **Align attention semantics**
- Add v_norm behavior for Nano attention.
- Add KV-shared-layer behavior based on model config (`num_kv_shared_layers`).

4. **Add parity harness before final text checks**
- Add deterministic micro-tests comparing intermediate tensors (single-layer outputs) against Transformers for a fixed seed/token sequence.
- Keep existing end-to-end semantic gate (`France -> Paris`) as final acceptance, not the only signal.

5. **Generalize hardcoded constants**
- Remove hardcoded `30` in per-layer embedding offset; use derived `num_layers` from model metadata.

## Sources

- HF Gemma 3n model implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3n/modeling_gemma3n.py
- HF Gemma 3n modular implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3n/modular_gemma3n.py
- HF Gemma 3n config implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3n/configuration_gemma3n.py
- HF model config (E2B): https://huggingface.co/google/gemma-3n-E2B-it/blob/main/config.json
- HF Gemma 3n docs: https://huggingface.co/docs/transformers/model_doc/gemma3n
- Gemma 3n model card: https://ai.google.dev/gemma/docs/core/model_card_3n
