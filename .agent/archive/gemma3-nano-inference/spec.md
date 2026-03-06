# Flow: gemma3-nano-inference

## Specification

### Code Analysis Summary

**Problem:** The Mojo backend (`model.mojo`, `layers.mojo`, `core.mojo`) only implements a standard transformer decoder. Gemma 3 Nano adds three architectural components that must be implemented in Mojo for inference:

1. **AltUp Routing (Mixture-of-Modalities):** Each layer routes its hidden state through 4 modality pathways using learned prediction and correction coefficients. This is the most complex addition.

2. **Laurel Pathway:** A parallel 64-dimensional bottleneck compression pathway (2048→64→2048) that provides auxiliary feature weighting.

3. **Per-Layer Embedding Mapping:** Each layer has access to a shared per-layer embedding table (262K×30×256) gated per-token, enabling adaptive feature injection.

**Architecture flow per Nano layer:**
```
Input (2048)
  ↓
PRE-ATTENTION NORM → ATTENTION → POST-ATTENTION NORM
  ↓
PER-LAYER MAPPING (gate → per-layer embed lookup → projection → norm)
  ↓
PRE-FFW NORM → MLP (gated) → POST-FFW NORM
  ↓
LAUREL (down 2048→64 → up 64→2048 → norm)
  ↓
ALTUP (router → 4-way prediction → correction → output scale)
  ↓
Output (2048)
```

### Files Affected

| File | Change Type | Description |
|------|-------------|-------------|
| `src/mo/mogemma/model.mojo` | Edit | Add `NanoLayerWeights`, `NanoModelWeights` structs with all Nano-specific fields |
| `src/mo/mogemma/layers.mojo` | Edit | Add `forward_altup`, `forward_laurel`, `forward_per_layer_mapping`, `forward_nano_layer` |
| `src/mo/mogemma/ops.mojo` | Edit (minor) | May need small helper ops for AltUp coefficient application |
| `src/mo/mogemma/core.mojo` | Edit | Add `_build_nano_model`, architecture-aware dispatch in `init_model_mojo`, `step_mojo`, `generate_embeddings_mojo` |
| `src/mo/tests/test_layers.mojo` | Edit | Add tests for Laurel, AltUp, per-layer mapping forward passes |
| `src/py/mogemma/model.py` | Edit (minor) | Ensure model type detection passes architecture flag to Mojo |

### Key Design Decisions

1. **Separate NanoLayerWeights vs extending LayerWeights:** Use a separate `NanoLayerWeights` struct that embeds a `LayerWeights` for the standard components and adds Nano-specific fields. Avoids bloating the standard path.

2. **Architecture dispatch:** `core.mojo` checks for Nano-specific tensors (e.g., `model.layers.0.altup.router.weight`) in metadata. If present, builds a `NanoModelWeights` and calls `forward_nano_layer`; otherwise uses the existing standard path.

3. **Per-layer embedding:** The massive 262K×30×256 tensor stays as a single contiguous block in memory. Access pattern: for layer L and token T, slice at `T * 30 * 256 + L * 256` for the 256-dim per-layer embedding.

4. **AltUp implementation:** The 4-way routing uses softmax over router logits, then applies prediction coefficients as a learned blend of the 4 projected variants, followed by correction step.

---

## Implementation Plan

### Phase 1: Nano Model Structs (src/mo/mogemma/model.mojo)

- [ ] 1.1 Create `AltUpWeights` struct with: router, router_norm, prediction_coefs, correction_coefs, output_scale as TensorInfo
- [ ] 1.2 Create `LaurelWeights` struct with: down_proj, up_proj, norm as TensorInfo
- [ ] 1.3 Create `PerLayerMapWeights` struct with: gate, projection, norm as TensorInfo
- [ ] 1.4 Create `NanoLayerWeights` struct that contains: `base: LayerWeights`, `altup: AltUpWeights`, `laurel: LaurelWeights`, `per_layer_map: PerLayerMapWeights`, plus `pre_feedforward_layernorm` and `post_feedforward_layernorm` norms
- [ ] 1.5 Create `NanoModelWeights` struct with: embed_tokens, norm, lm_head, per_layer_embed (TensorInfo for the 3D tensor), per_layer_projection, per_layer_norm, altup_projections (List), altup_unembeds (List), layers (List[NanoLayerWeights])

### Phase 2: Nano Forward Pass Components (src/mo/mogemma/layers.mojo)

- [ ] 2.1 Implement `forward_per_layer_mapping(out_ptr, x_ptr, layer_idx, token_id, per_layer_embed_ptr, weights: PerLayerMapWeights, hidden_size, per_layer_dim)` — gate input, lookup per-layer embedding, project back, add residual
- [ ] 2.2 Implement `forward_laurel(out_ptr, x_ptr, weights: LaurelWeights, hidden_size, bottleneck_dim, scratch_ptr)` — down project, up project, norm, add residual
- [ ] 2.3 Implement `forward_altup_predict(out_ptr, variants_ptr, weights: AltUpWeights, hidden_size, num_modalities)` — apply prediction coefficients across 4 modality variants
- [ ] 2.4 Implement `forward_altup_correct(out_ptr, predicted_ptr, weights: AltUpWeights, hidden_size, num_modalities)` — apply correction coefficients and output scaling
- [ ] 2.5 Implement `forward_altup(out_ptr, x_ptr, altup_projections, altup_unembeds, weights: AltUpWeights, hidden_size, scratch_ptr)` — full AltUp routing pipeline: project variants → router → predict → correct → unembed → output
- [ ] 2.6 Implement `forward_nano_layer(out_ptr, x_ptr, weights: NanoLayerWeights, ...)` — full Nano layer: attention → per-layer map → MLP → Laurel → AltUp

### Phase 3: Architecture-Aware Core (src/mo/mogemma/core.mojo)

- [ ] 3.1 Add `_build_nano_model(metadata_obj) -> NanoModelWeights` that loads all 671 tensors
- [ ] 3.2 Add `_detect_architecture(metadata_obj) -> String` that checks for Nano-specific tensor names
- [ ] 3.3 Modify `init_model_mojo` to detect architecture, build appropriate model, store architecture flag in llm dict
- [ ] 3.4 Modify `step_mojo` to dispatch to standard or Nano forward pass based on architecture flag
- [ ] 3.5 Modify `generate_embeddings_mojo` to dispatch to standard or Nano forward pass
- [ ] 3.6 Handle Nano's different KV cache layout (2 KV heads vs variable)

### Phase 4: Tests

- [ ] 4.1 Add unit tests for `forward_laurel` with known input/output (2048→64→2048 bottleneck)
- [ ] 4.2 Add unit tests for `forward_per_layer_mapping` with synthetic per-layer embeddings
- [ ] 4.3 Add unit tests for `forward_altup` with 4-way routing coefficients
- [ ] 4.4 Add unit tests for `forward_nano_layer` (full layer)
- [ ] 4.5 Add Python integration test: load converted gemma3n-e2b-it and verify init_model succeeds
- [ ] 4.6 Add Python integration test: single forward step produces valid logits (correct vocab size)
- [ ] 4.7 Run full test suite — no regressions in standard Gemma 3 path

### Phase 5: End-to-End Validation

- [ ] 5.1 Update `scripts/validate.py` to support `--model` flag for arbitrary model selection
- [ ] 5.2 Add `gemma3n-e2b-it` as a validation target (once Mojo backend supports it)
- [ ] 5.3 Verify text generation produces coherent output
- [ ] 5.4 Verify embedding generation produces valid embeddings
- [ ] 5.5 Performance baseline: compare Nano vs standard Gemma 3 inference speed
