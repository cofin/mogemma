# Flow: gemma3-nano-conversion

## Specification

### Code Analysis Summary

**Problem:** The existing `_convert_gemma3()` in `convert.py` handles standard Gemma 3 models (~237-341 tensors) but Gemma 3 Nano has **671 tensors** with a fundamentally different architecture. A new conversion function is needed.

**Nano Architecture (from OCDBT analysis of gemma3n-e2b-it):**

| Category | Tensors | Description |
|----------|---------|-------------|
| Embedder | 4 | Standard + per-layer embeddings (262K×30×256) |
| AltUp Global | 6 | 3 projection + 3 unembed matrices (2048×2048) |
| AltUp Per-Layer | 150 | Router, prediction/correction coefficients (5 per layer) |
| Per-Layer Mapping | 90 | Gate, projection, norm (3 per layer) |
| Laurel | 90 | Down/up projection + norm (3 per layer) |
| Attention | 150 | Q/KV/O projections + QK norms (5 per layer) |
| MLP | 60 | Linear + gating_einsum (2 per layer) |
| Norms | 121 | Pre/post attention + pre/post FFW + final (4 per layer + 1) |
| **Total** | **671** | |

**Model dimensions:** hidden=2048, heads=8, kv_heads=2, head_dim=256, intermediate=8192, layers=30

**Key differences from standard Gemma 3:**
1. `embedder.per_layer_input_embedding` [262144, 30, 256] — massive per-layer token embedding table
2. `embedder.per_layer_input_projection.w` [2048, 30, 256] — projection from hidden to per-layer space
3. 5 AltUp tensors per layer (router, prediction/correction coefficients, output scale, router norm)
4. 3 Laurel tensors per layer (64-dim bottleneck compression pathway)
5. 3 per-layer mapping tensors (gate, projection, post-norm)
6. Attention uses `attn.query_norm` (not `attn._query_norm` like standard)
7. MLP has `mlp.linear` (not `mlp.linear.w` — check exact naming)

### Files Affected

| File | Change Type | Description |
|------|-------------|-------------|
| `src/py/mogemma/convert.py` | Edit | Add `_convert_gemma3_nano()` function, add `_detect_model_family()` dispatcher |
| `src/py/mogemma/hub.py` | Edit | Update `_ensure_safetensors` to use family-aware conversion |
| `src/py/tests/test_convert.py` | New | Unit tests for Nano tensor mapping |

### Nano Tensor Naming Map (Orbax → HF Safetensors)

**Global tensors:**
```
embedder.input_embedding → model.embed_tokens.weight + lm_head.weight
embedder.per_layer_input_embedding → model.per_layer_embed.weight
embedder.per_layer_input_projection.w → model.per_layer_embed.projection.weight
embedder.per_layer_projection_norm.scale → model.per_layer_embed.norm.weight
final_norm.scale → model.norm.weight
altup_projection.{i} → model.altup.projection.{i}.weight
altup_unembed_projection.{i} → model.altup.unembed.{i}.weight
```

**Per-layer tensors (layer N):**
```
# Standard (same transforms as _convert_gemma3)
pre_attention_norm.scale → model.layers.{N}.input_layernorm.weight
post_attention_norm.scale → model.layers.{N}.post_attention_layernorm.weight
pre_ffw_norm.scale → model.layers.{N}.pre_feedforward_layernorm.weight
post_ffw_norm.scale → model.layers.{N}.post_feedforward_layernorm.weight
attn.q_einsum.w → model.layers.{N}.self_attn.q_proj.weight (transpose+reshape)
attn.kv_einsum.w → model.layers.{N}.self_attn.k_proj.weight + v_proj.weight (split+transpose+reshape)
attn.attn_vec_einsum.w → model.layers.{N}.self_attn.o_proj.weight (reshape+transpose)
attn.query_norm.scale → model.layers.{N}.self_attn.q_norm.weight
attn.key_norm.scale → model.layers.{N}.self_attn.k_norm.weight
mlp.gating_einsum → model.layers.{N}.mlp.gate_proj.weight + up_proj.weight
mlp.linear → model.layers.{N}.mlp.down_proj.weight (transpose)

# Nano-specific: AltUp
altup.modality_router.w → model.layers.{N}.altup.router.weight
altup.router_norm_layer.scale → model.layers.{N}.altup.router_norm.weight
altup.prediction_coefs → model.layers.{N}.altup.prediction_coefs
altup.correction_coefs → model.layers.{N}.altup.correction_coefs
altup.correct_output_scale → model.layers.{N}.altup.output_scale

# Nano-specific: Per-layer mapping
per_layer_mapping.per_layer_input_gate.w → model.layers.{N}.per_layer_map.gate.weight
per_layer_mapping.per_layer_projection.w → model.layers.{N}.per_layer_map.projection.weight
per_layer_mapping.post_per_layer_input_norm.scale → model.layers.{N}.per_layer_map.norm.weight

# Nano-specific: Laurel
laurel.linear_left.w → model.layers.{N}.laurel.down_proj.weight
laurel.linear_right.w → model.layers.{N}.laurel.up_proj.weight
post_laurel_norm.scale → model.layers.{N}.laurel.norm.weight
```

---

## Implementation Plan

### Phase 1: Model Family Detection

- [x] 1.1 Add `_detect_model_family(orbax_keys: list[str]) -> str` function to `convert.py` that returns `"gemma3"` or `"gemma3_nano"` based on presence of `altup` or `per_layer_mapping` tensors
- [x] 1.2 Update `convert_orbax_to_safetensors()` to call `_detect_model_family()` and dispatch to the right converter
- [x] 1.3 Add `_is_nano_checkpoint(path: Path) -> bool` utility for hub.py to check before conversion

### Phase 2: Nano Conversion Function

- [x] 2.1 Create `_convert_gemma3_nano(orbax: dict) -> dict` function with full tensor mapping
- [x] 2.2 Implement global tensor conversion (embedder, AltUp projections, final norm)
- [x] 2.3 Implement per-layer standard tensor conversion (attention, MLP, norms — shared logic with `_convert_gemma3`)
- [x] 2.4 Implement per-layer AltUp tensor conversion (router, coefficients, output scale)
- [x] 2.5 Implement per-layer mapping tensor conversion (gate, projection, norm)
- [x] 2.6 Implement per-layer Laurel tensor conversion (down/up projections, norm)
- [x] 2.7 Handle the 3D per_layer_input_embedding tensor (262144×30×256) — determine safetensors storage strategy (single tensor or split per-layer)
- [x] 2.8 All conversions must cast to float32 and ensure contiguous layout

### Phase 3: Shared Conversion Logic Refactor

- [x] 3.1 Extract common attention conversion logic from `_convert_gemma3` and `_convert_gemma3_nano` into `_convert_attention_layer()` helper (Q transpose+reshape, KV split, O reshape+T)
- [x] 3.2 Extract common MLP conversion logic into `_convert_mlp_layer()` helper (gating split, linear transpose)
- [x] 3.3 Extract common norm conversion into `_convert_norms()` helper
- [x] 3.4 Refactor both `_convert_gemma3` and `_convert_gemma3_nano` to use shared helpers

### Phase 4: Hub Integration

- [x] 4.1 Update `hub.py` `_get_tokenizer_path` to ensure Nano tokenizer is correctly resolved
- [x] 4.2 Verify `_ensure_safetensors` works for Nano checkpoints end-to-end
- [x] 4.3 Add logging for model family detection during conversion

### Phase 5: Tests

- [x] 5.1 Create `test_convert.py` with unit tests for `_detect_model_family()`
- [x] 5.2 Add tests for shared conversion helpers (attention, MLP, norm)
- [x] 5.3 Add test: Nano conversion produces expected tensor count and names
- [x] 5.4 Add test: round-trip Nano conversion preserves shapes and dtypes
- [x] 5.5 Integration test: convert real gemma3n-e2b-it OCDBT checkpoint (skip if not cached)
- [x] 5.6 Run full test suite + linting
