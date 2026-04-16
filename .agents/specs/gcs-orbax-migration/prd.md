# Master PRD: GCS Download + Orbax→Safetensors Migration

## Context (North Star)

Replace the HuggingFace HTTP download backend with direct downloads from Google's
public `gs://gemma-data` bucket. The GCS bucket distributes Gemma 4 weights in
**Orbax/OCDBT format** (JAX native), so a one-time conversion to safetensors is
required before the existing Mojo FFI bridge can consume them.

Additionally, the **EmbeddingConfig** should default to pre-trained (`-pt`) models
rather than instruction-tuned (`-it`), since PT models produce higher-quality
embeddings (IT fine-tuning optimizes for instruction-following, not representation
learning).

### Why

- HuggingFace downloads have been consistently unreliable for scripted/automated use.
- The `gemma-data` GCS bucket is public, requires no authentication (`skip_signature`),
  and contains all 8 Gemma 4 model variants (4 families × IT/PT).
- Eliminating the HF dependency simplifies the download pipeline and removes a
  fragile external dependency.

### Key Outcomes

1. `HubManager` downloads from GCS instead of HuggingFace.
2. Orbax checkpoints are converted to safetensors and cached locally; Orbax files
   are cleaned up after conversion to minimize disk usage.
3. All 4 Gemma 4 families supported: Dense-31B, E2B, E4B, MoE-26B-A4B (both IT and PT).
4. `EmbeddingConfig` defaults to the `-pt` variant; `GenerationConfig` keeps `-it`.
5. All HuggingFace-specific code paths removed.

---

## GCS Bucket Structure

**Bucket:** `gemma-data` (public, no auth)

**Checkpoints:** `checkpoints/{clean_model_id}/` — Orbax/OCDBT format  
**Tokenizers:** `tokenizers/tokenizer_gemma4.model` — shared across all Gemma 4 variants

### Available Checkpoints

| Variant       | IT path                          | PT path                          |
|---------------|----------------------------------|----------------------------------|
| Dense 31B     | `checkpoints/gemma4-31b-it/`     | `checkpoints/gemma4-31b-pt/`     |
| E2B           | `checkpoints/gemma4-e2b-it/`     | `checkpoints/gemma4-e2b-pt/`     |
| E4B           | `checkpoints/gemma4-e4b-it/`     | `checkpoints/gemma4-e4b-pt/`     |
| MoE 26B-A4B   | `checkpoints/gemma4-26b-a4b-it/` | `checkpoints/gemma4-26b-a4b-pt/` |

---

## Orbax Tensor Name Discovery

Full tensor trees were extracted from the live GCS checkpoints. Below are the
unique tensor templates per variant (layer indices collapsed to `N`).

### Base Transformer (all variants)

| Orbax Name | Notes |
|---|---|
| `embedder.input_embedding` | Token embeddings (tied to LM head) |
| `final_norm.scale` | Final RMSNorm |
| `layer_N.attn.q_einsum.w` | Query projection (einsum layout) |
| `layer_N.attn.kv_einsum.w` | Combined KV projection (E2B/E4B) |
| `layer_N.attn.k_einsum.w` | Separate K projection (31B/MoE) |
| `layer_N.attn.attn_vec_einsum.w` | Output projection |
| `layer_N.attn.query_norm.scale` | Query RMSNorm |
| `layer_N.attn.key_norm.scale` | Key RMSNorm |
| `layer_N.mlp.gating_einsum.w` | Combined gate+up projection |
| `layer_N.mlp.linear.w` | Down projection |
| `layer_N.pre_attention_norm.scale` | Pre-attn norm |
| `layer_N.post_attention_norm.scale` | Post-attn norm |
| `layer_N.pre_ffw_norm.scale` | Pre-FFW norm |
| `layer_N.post_ffw_norm.scale` | Post-FFW norm |
| `layer_N.skip_scale` | Skip connection scaling factor |

### E2B/E4B Extras (Per-Layer Embeddings)

| Orbax Name | Notes |
|---|---|
| `embedder.per_layer_embeddings` | Shared PLE embedding table |
| `embedder.per_layer_model_projection.w` | Model projection for PLE |
| `embedder.per_layer_projection_norm.scale` | PLE projection norm |
| `layer_N.per_layer_input_gate.w` | Per-layer gating |
| `layer_N.per_layer_projection.w` | Per-layer projection |
| `layer_N.post_per_layer_input_norm.scale` | Post-PLE norm |

### E2B Audio-specific

| Orbax Name | Notes |
|---|---|
| `embedder.audio_input_embedding_extra` | Audio embedding extra tokens |
| `embedder.audio_input_projection.w` | Audio→text projection |

### Multimodal (all variants with vision)

| Orbax Name | Notes |
|---|---|
| `embedder.mm_input_embedding_extra` | Vision embedding extra tokens |
| `embedder.mm_input_projection.w` | Vision→text projection |

### Vision Encoder (all variants)

| Orbax Name | Notes |
|---|---|
| `vision_encoder.entry.input_projection.w` | Patch embedding projection |
| `vision_encoder.entry.pos_emb` | Positional embeddings |
| `vision_encoder.standardize.{scale,bias}` | Input standardization (31B/MoE only) |
| `vision_encoder.transformer.stacked_layers.block.attn.q_einsum.w` | Vision Q proj |
| `vision_encoder.transformer.stacked_layers.block.attn.kv_einsum.w` | Vision KV proj |
| `vision_encoder.transformer.stacked_layers.block.attn.attn_vec_einsum.w` | Vision O proj |
| `vision_encoder.transformer.stacked_layers.block.attn.query_norm.scale` | Vision Q norm |
| `vision_encoder.transformer.stacked_layers.block.attn.key_norm.scale` | Vision K norm |
| `vision_encoder.transformer.stacked_layers.block.mlp.gating_einsum.w` | Vision gate+up |
| `vision_encoder.transformer.stacked_layers.block.mlp.linear.w` | Vision down proj |
| `vision_encoder.transformer.stacked_layers.block.{pre,post}_attention_norm.scale` | Vision attn norms |
| `vision_encoder.transformer.stacked_layers.block.{pre,post}_ffw_norm.scale` | Vision FFW norms |

### MoE-specific (26B-A4B)

| Orbax Name | Notes |
|---|---|
| `layer_N.mlp.router_logits.w` | Expert routing gating network |
| `layer_N.mlp.router_scale` | Router scaling factor |
| `layer_N.mlp.per_expert_scale` | Per-expert scaling factor |
| `layer_N.mlp2.gating_einsum.w` | Second MLP gating (MoE expert) |
| `layer_N.mlp2.linear.w` | Second MLP down projection |
| `layer_N.pre_ffw2_norm.scale` | Pre-FFW2 norm |
| `layer_N.post_ffw1_norm.scale` | Post-FFW1 norm |
| `layer_N.post_ffw2_norm.scale` | Post-FFW2 norm |

### Audio Encoder (E2B — Conformer architecture)

The audio encoder uses a **Conformer** architecture (12 stacked layers), fundamentally
different from the ViT-style vision encoder. Each conformer layer contains:

| Orbax Name Pattern (per stacked_layers_N) | Component |
|---|---|
| `fflayer_start.ffn_layer{1,2}.kernel` | Start feed-forward block |
| `fflayer_start.{pre,post}_layer_norm.scale` | Start FF norms |
| `trans_atten.self_atten.{query,key,value}.kernel` | Multi-head self-attention |
| `trans_atten.self_atten.per_dim_scale` | Per-dimension attention scaling |
| `trans_atten.self_atten.relative_position_embedding.pos_proj.kernel` | Relative PE |
| `trans_atten.post.kernel` | Attention output projection |
| `trans_atten.{pre,post}_norm.scale` | Attention norms |
| `lconv.depthwise_conv1d.kernel` | Lightweight depthwise convolution |
| `lconv.linear_{start,end}.kernel` | Conv gating projections |
| `lconv.{conv_norm,ln}.scale` | Conv norms |
| `fflayer_end.ffn_layer{1,2}.kernel` | End feed-forward block |
| `fflayer_end.{pre,post}_layer_norm.scale` | End FF norms |
| `final_ln.scale` | Per-layer final norm |

**Audio feature extraction:**

| Orbax Name | Notes |
|---|---|
| `audio_encoder.feature.input_proj.kernel` | Mel-spectrogram input projection |
| `audio_encoder.feature.norm_{0,1}.scale` | Feature normalization layers |
| `audio_encoder.feature.subsampling_{0,1}.kernel` | Downsampling conv layers |
| `audio_encoder.output_projection.{kernel,bias}` | Audio output projection |

**Note:** Audio tensors also have `clip_input_{min,max}` and `clip_output_{min,max}`
per-kernel for quantization ranges. These should be converted as `*.weight_scale` or
similar based on what the Mojo backend expects.

---

## Roadmap

### Chapter 1: `gcs-download-backend`
**Replace HuggingFace download with GCS + OrbaxLoader**

Rewrite `HubManager` to download from `gs://gemma-data` using `obstore.store.GCSStore`.
Resurrect and modernize `OrbaxLoader` for Gemma 4. Handle file enumeration via
`obs.list()` instead of index.json parsing. Download tokenizer from
`tokenizers/tokenizer_gemma4.model`.

### Chapter 2: `orbax-safetensors-conversion`
**Convert Orbax tensors to HuggingFace-style safetensors**

Write the tensor name mapping and shape transformation logic for all 4 Gemma 4
variants: base transformer, vision encoder, audio encoder (Conformer), MoE routing,
and Per-Layer Embeddings. Convert and cache safetensors locally. Clean up Orbax files
after successful conversion to minimize disk usage. Generate `config.json` from
checkpoint metadata.

### Chapter 3: `pt-embedding-autoselect`
**Auto-select pre-trained models for embeddings**

Change `EmbeddingConfig.model_path` default to `google/gemma-4-26B-A4B-pt`. Add
logic in `HubManager` or model init to detect embedding vs generation context and
select the appropriate tuning variant.

### Chapter 4: `hf-removal-cleanup`
**Remove all HuggingFace code and update tests**

Delete HF-specific code: `_make_hf_store`, `_get_hf_token`, `_hf_resolve_url`,
`HTTPStore` imports, `_fetch_index_json`, `_parse_shard_filenames`, and all
HF-related tests. Update error messages, docstrings, and README. Run full test
suite to verify.

---

## Global Constraints

1. **No HuggingFace dependency** — all download paths must use GCS.
2. **Disk space conservation** — Orbax files deleted after successful safetensors conversion.
3. **obstore-based** — use `obstore.store.GCSStore` (already a project dependency).
4. **Safetensors naming must match Mojo backend** — the tensor names in converted
   safetensors must exactly match what `core.mojo` loads via `SafetensorsLoader`.
5. **Public bucket** — `GCSStore("gemma-data", config={"skip_signature": "true"})`,
   no authentication required.
6. **Tokenizer shared** — all Gemma 4 variants use `tokenizers/tokenizer_gemma4.model`.
