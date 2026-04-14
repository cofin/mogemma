# Gemma 4 model architectures

Reference for the four supported variants and their architectural differences.
Tensor-name details and shape contracts live in component files; this doc is
the conceptual map.

## Variants

| Variant | Params | Active | Context | Modalities | Distinctive features |
|---|---|---|---|---|---|
| **E2B** | ~2B | 2B | 128K | text + image + audio | Dense, PLE (per-layer embeddings) |
| **E4B** | ~4B | 4B | 128K | text + image + audio | Dense, PLE |
| **31B** | 31B | 31B | 256K | text + image | Dense, no PLE |
| **26B-A4B** | 26B | ~4B | 256K | text + image | MoE (128 experts, top-8) + dense branch |

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

```
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

## Related

- [python-runtime.md](python-runtime.md) — loaders + convert.py
- [mojo-runtime.md](mojo-runtime.md) — struct layouts
