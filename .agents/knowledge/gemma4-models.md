# Gemma 4 model architectures

Reference for the official Gemma 4 variants and their architectural differences.
Tensor-name details and shape contracts live in component files; this doc is
the conceptual map.

## Variants

| Variant | Params | Active | Context | Modalities | Distinctive features |
|---|---|---|---|---|---|
| **E2B** (`DENSE_E2B`) | ~2B | 2B | 128K | text + image + audio | Dense, PLE, double-wide MLP (4x) |
| **E4B** (`DENSE_E4B`) | ~4B | 4B | 128K | text + image + audio | Dense, PLE, standard MLP (8x) |
| **12B** (`DENSE_12B_UNIFIED`) | 12B | 12B | 256K | text + image + audio | Dense unified encoder-free multimodal architecture; local CPU text runtime supported, unified image/audio gated |
| **31B** (`DENSE_31B`) | 31B | 31B | 256K | text + image | Dense, no PLE |
| **26B-A4B** (`MOE_26B_A4B`) | 26B | ~4B | 256K | text + image | MoE (128 experts, top-8) + dense branch |

Runtime support is tracked separately from official model status in
`src/py/mogemma/model_support.py`. A model appearing in this table means it is
an official Gemma 4 architecture, not that every modality is implemented in the
local Mojo runtime.

## Core architecture (all variants)

- Transformer decoder with **hybrid attention**: sliding-window layers alternate with full-attention layers. Periodic full-attn at layers 6, 12, 18, 24, 30 (1-indexed in HF config).
- RoPE: θ=10000 on sliding layers, θ=1000000 on full layers with partial rotation.
- RMSNorm with `with_scale=True` on all pre/post norms except router's internal norm.
- GEGLU MLP: `gate, up = fc1(x).chunk(2); h = gelu(gate) * up; out = fc2(h)`.
- Tied embeddings: `lm_head.weight = embed_tokens.weight`.

## Attention details

- Per-head `q_norm` and `k_norm` applied before RoPE.
- Sliding window size set via `sliding_window` config key.
- `kv_einsum.w` packs K+V when `attention_k_eq_v=True` (local attn); separate `k_einsum.w` / `v_einsum.w` when they differ (global attn layers). Both shapes exist in the same checkpoint — inspect per-layer.
- Shared-KV layers: some layers reuse prior-layer K/V via `kv_sharing_layer_map` in config. Forward-pass skips K/V projection for these layers.

## PLE (Per-Layer Embeddings) — E2B / E4B only

- At each layer start, token id is also looked up in a per-layer embedding table `(V, L, H_ple)`, projected to `H` via a bottleneck, normed, and **added** to the hidden state.
- Per-layer tensors: `embedder.per_layer_embeddings[:, N, :]`, `layer_N.per_layer_projection.w`, `layer_N.post_per_layer_input_norm.scale`.
- Non-contract Orbax tensors to skip during conversion: `embedder.per_layer_model_projection.w`, `embedder.per_layer_projection_norm.scale`, `layer_N.per_layer_input_gate.w`, `layer_N.skip_scale` (on PLE variants; NOT on MoE where skip_scale has meaning).
- PLE is a **bottleneck augmentation**, not a replacement for the main residual path.

## MoE (26B-A4B) — two-branch architecture

Critical: this is NOT the HuggingFace `block_sparse_moe` pattern. Every layer
has **both** a dense MLP (`mlp2`) and a routed MoE (`mlp`); their outputs are
summed. No layer-skipping: all 30 layers are MoE layers.

### Forward pass

```
h1 = post_ffw1_norm(dense_mlp(pre_ffw_norm(x)))      # dense branch
x_r = pre_ffw2_norm(x)
_, topk_w, topk_idx = router(x)                       # router consumes pre-norm x
h2 = post_ffw2_norm(experts(x_r, topk_idx, topk_w))   # MoE branch
out = residual + h1 + h2                              # (+ skip_scale contribution — see below)
```

### Router

```mojo
r = RMSNorm(x, no_scale) * router_scale * hidden_size^-0.5
probs = softmax(router_logits @ r)
topk_w, topk_idx = topk(probs, k=num_experts_per_tok)   # k=8
topk_w /= topk_w.sum(-1, keepdim=True)                   # renormalize
topk_w *= per_expert_scale[topk_idx]                     # per-expert gain
```

### Experts (GEGLU per selected expert)

```
gate, up = (gate_up_proj[e] @ x_r).chunk(2, -1)
h = gelu(gate) * up
out_e = down_proj[e] @ h
final[token] += out_e * topk_w[token, pos]
```

### Shapes (26B-A4B: H=2816, I_moe=704, I_dense=2112, E=128)

| Tensor | Shape |
|---|---|
| `layer_N.mlp.router_logits.w` | `(2816, 128)` = `(H, E)`; convert to `(E, H)` via `.T` |
| `layer_N.mlp.router_scale` | `(2816,)` |
| `layer_N.mlp.per_expert_scale` | `(128,)` |
| `layer_N.mlp.gating_einsum.w` | `(128, 2, 704, 2816)` → reshape to `(E, 2·I_moe, H)` |
| `layer_N.mlp.linear.w` | `(128, 704, 2816)` → transpose to `(E, H, I_moe)` |
| `layer_N.mlp2.gating_einsum.w` | `(2, 2112, 2816)` → split to dense gate/up |
| `layer_N.mlp2.linear.w` | `(2112, 2816)` → `.T` for dense down_proj |

### Per-layer norms (six total on MoE layers)

- `pre_attention_norm`, `post_attention_norm`
- `pre_ffw_norm` (dense pre) / `post_ffw1_norm` (dense post, h1)
- `pre_ffw2_norm` (MoE pre) / `post_ffw2_norm` (MoE post, h2)
- `post_ffw_norm` — present alongside post_ffw1/post_ffw2; role unclear, emit faithfully

### `skip_scale` (1,) per layer

Scalar not in HF reference code. May apply to residual (`residual * skip_scale`).
Inspect live checkpoint value before wiring into Mojo forward.

## Vision encoder (all multimodal variants)

- SigLIP-derived, bidirectional (non-causal) attention, standard GELU.
- Fixed-budget patch count with variable aspect ratio support. H and W must be divisible by 48 (patch 16 × pool 3).
- Patch embedding is a conv2d `(H_v, C, patch, patch)` reshaped from Orbax `(patch²·C, H_v)`.
- MLP uses **GEGLU**, not plain MLP. Vision `gating_einsum` splits into `fc1` + `fc1_up`; Mojo `VisionLayerWeights` must include both.
- Does **not** apply ImageNet mean/std; patch embedding scales to `[-1, 1]` internally.
- Orbax packs all vision layers via vmap: `ve.stacked_layers.block.*[I, ...]` → split along leading axis to per-layer.

## Audio encoder (E2B / E4B only)

- Reuses vision transformer structure.
- Mel spectrograms computed in Python (`audio.py`), passed into FFI as float32 tensors.
- Audio/image/video special tokens: IDs 258880–258884.
- Current runtime status: audio input is recognized but deliberately rejected.
  `process_audio_mojo()` raises a clear unsupported-runtime error instead of
  appending zero embeddings.

## 12B unified multimodal architecture

- The official 12B release uses `model_type = "gemma4_unified"` with
  `architectures = ["Gemma4UnifiedForConditionalGeneration"]` and nested
  `text_config.model_type = "gemma4_unified_text"`.
- It is not the same shape as the existing E2B/E4B/31B split-encoder runtime.
  Image patches and audio frames are fed through a unified encoder-free model
  path, so the existing `vision_config`/`audio` side-tower assumptions cannot
  simply be reused.
- Mogemma recognizes this architecture in `_detect_gemma4_variant()` and
  supports local-safetensors CPU text initialization after the dedicated 12B
  config and variable-head runtime contracts pass.
- `google/gemma-4-12B-it` is not listed in `KNOWN_GCS_MODELS` until a live GCS
  probe proves checkpoint availability. Remote download and Orbax conversion
  remain gated without a validated 12B tensor inventory.
- Unified image/audio inputs and GPU variable-head attention are explicitly
  rejected until follow-up runtime work lands.

## KV cache

- **Single contiguous arena** for all layers (preferred over per-layer allocation — reduces host↔device traffic and allocation overhead).
- Ring buffer for windowed attention.
- Shared-KV layers skip the write to avoid double-booking.

## Config keys consumed by Mojo

Validated at `model.py:109-203`:
- `sliding_window_size`, `partial_rotary_factor`, `attention_k_eq_v`
- `layer_types` — list of `"sliding"` / `"full"`
- `vision_config.{num_hidden_layers, hidden_size, num_attention_heads, intermediate_size}`
- `image_token_index`, `audio_token_index`
- `hidden_size_per_layer_input`, `vocab_size_per_layer_input` (PLE)
- `use_double_wide_mlp`, `kv_sharing_layer_map`
- `num_local_experts` / `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`

## MoE layer — skip_scale (26B-A4B-it)

**Source:** empirical probe against `google/gemma-4-26B-A4B-it` Orbax
checkpoint, 2026-04-16. See `gemma4-26b-skip-scale.csv` for per-layer
values.

**Decision: LIVE — wire `x1 * skip_scale[0]` into the MoE residual sum.**

All 30 layers have positive, substantive values:

- min: +0.070530 (layer 0)
- max: +0.815137 (layer 24)
- mean: +0.624398
- std: 0.196567
- count `|v| < 1e-6`: 0 / 30

The distribution shows a characteristic training-learned pattern —
smaller at the boundary layers (0, 29), settling near ~0.7 in the
mid-stack. Values are never zero and never negative; this is a
residual-amplification scalar, not a gate.

**Forward-pass implication (CORRECTED 2026-04-16):** This tensor is
HF's `layer_scalar` — a **multiplicative scalar on the entire layer
output**, applied as the last operation in
`Gemma4TextDecoderLayer.forward()`:

```python
hidden_states *= self.layer_scalar
return hidden_states
```

It is NOT added to the residual. Orbax `layer_N.skip_scale` →
converter output `model.layers.N.moe_skip_scale.weight` → HF
`layers.{N}.layer_scalar`. All three names refer to the same `(1,)`
buffer. See moe-mojo-runtime spec Phase 4.1 for wiring details.

## MoE layer — post_feedforward_layernorm (26B-A4B-it)

**Source:** empirical probe against `google/gemma-4-26B-A4B-it` Orbax
checkpoint, 2026-04-16. See `gemma4-26b-post-ffw-norm.csv` for
per-layer weight statistics.

**Decision: LIVE — norm is significantly non-identity, placement
hypothesis must still be enumerated (Phase 0.1c).**

Per-layer `||w − 1||_inf` across all 30 layers:

- max: 21.950363 (layer 0)
- layer 29: 8.950735
- mid-stack (layers 4–28): range 0.97 – 2.71
- min (closest to identity): 0.969750 (layer 11) — still well above
  any identity threshold

Per-layer weight-mean distribution:

- layer 0: +13.75 (boundary — strong FFN-output amplification)
- layers 4–28: +0.87 to +2.42 (mid-stack moderate)
- layer 29: +5.83 (boundary — strong again)

This "boundary-layer extreme, mid-stack moderate" curve is not a
quirk of initialization — it's a learned training dynamic, which
means the tensor is applied somewhere in the forward pass. The
rule-out of hypothesis H-C (vestigial) is definitive.

**Cross-check of per-branch norms (layer 0):**

- `post_ffw1_norm.scale`: mean +4.34, `||w − 1||_inf` = 88.5
- `post_ffw2_norm.scale`: mean +4.41, `||w − 1||_inf` = 134.1

All three norms (post_ffw1, post_ffw2, post_ffw) are live and
independently learned. This rules out H-B (single global norm
replacing per-branch) — if post_ffw_norm subsumed the per-branch
norms, the per-branch norms would be identity. They aren't.

**Placement CONFIRMED 2026-04-16 as H-A** via direct reading of
HuggingFace `transformers/src/transformers/models/gemma4/modeling_gemma4.py`
`Gemma4TextDecoderLayer.forward()`:

```python
hidden_states = hidden_states_1 + hidden_states_2         # h1 + h2
hidden_states = self.post_feedforward_layernorm(hidden_states)
hidden_states = residual + hidden_states                  # x1 + ...
```

H-B and H-D are ruled out by direct source reading; no probe harness
was needed.

## Vision encoder — note

The Orbax inventory contains 31 `post_ffw_norm` keys, not 30: the
extra one is
`vision_encoder.transformer.stacked_layers.block.post_ffw_norm.scale`
— part of the vision tower, not the MoE transformer.

## Related

- [python-runtime.md](python-runtime.md) — loaders + convert.py
- [mojo-runtime.md](mojo-runtime.md) — struct layouts
- [gemma4-26b-skip-scale.csv](gemma4-26b-skip-scale.csv) — per-layer
  skip_scale probe data
- [gemma4-26b-post-ffw-norm.csv](gemma4-26b-post-ffw-norm.csv) —
  per-layer post_ffw_norm weight statistics
