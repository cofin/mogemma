# Chapter 7: 26B A4B MoE Runtime

**Flow ID:** `gemma4-moe`
**Parent PRD:** `gemma4-rewrite`
**Beads Epic:** `mogemma-kuil.7`
**Depends on:** Chapter 3 (attention engine), Chapter 5 (vision pipeline)
**Status:** Planned

---

## Goal

Implement Mixture-of-Experts execution for the Gemma 4 26B A4B variant: 128 experts with top-8 routing, sparse expert dispatch, and the full forward pass. 26B A4B supports text + vision (no audio) with a 27-layer vision encoder. It uses K=V attention and 2 global KV heads.

## Current Codebase State (Post-Ch4)

**Reusable:**
- `forward_sliding_attention` / `forward_full_attention` — already support `k_eq_v: Bool` flag. K=V attention is **already implemented**.
- `forward_gemma4_layer` — dispatches sliding/full based on `kv_cache.layer_types[layer_idx]`.
- `KVCache` — hybrid sliding/full with configurable per-layer types. Works for 26B's 1024-token window.
- `RoPETables` — dual theta (10K/1M). Works as-is.
- `geglu` in ops.mojo — each MoE expert uses GEGLU MLP.
- `softmax` — needed for router probability normalization.
- `_detect_gemma4_variant` already detects `MOE_26B_A4B` via `num_experts > 0`.

**Missing:**
- No MoE weight structs (expert weights, router)
- No top-k selection op
- No expert routing or sparse dispatch
- No MoE layer forward function

## Architecture Reference

### MoE Layer Structure
```
input → input_layernorm → attention (K=V, 2 KV heads, hybrid sliding/full)
      → residual + post_attention_layernorm
      → router(hidden) → softmax → top-8 selection
      → 8 expert MLPs (gate→GEGLU←up → down) at intermediate=704
      → weighted sum of expert outputs
      → residual
```

### 26B A4B Parameters

| Parameter | Value |
|---|---|
| Decoder layers | 30 |
| Hidden size | ~3584 (from config.json) |
| Attention heads | (from config.json) |
| KV heads | 2 (global) |
| `attention_k_eq_v` | true |
| Num experts | 128 |
| Top-k | 8 |
| `moe_intermediate_size` | 704 |
| Vision encoder | 27 layers (from Ch5) |
| Audio | No |
| Sliding window | 1024 |
| Context | 256K |

### Expert Weight Layout (HF Tensor Names)
```
model.layers.{i}.block_sparse_moe.gate.weight              [128, hidden_size]  (router)
model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight    [704, hidden_size]  (gate_proj)
model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight    [hidden_size, 704]  (down_proj)
model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight    [704, hidden_size]  (up_proj)
```

j = 0..127 per layer. Total: 30 layers × 128 experts × 3 projections = 11,520 weight tensors for MoE alone.

### Memory

- Per expert per layer: 3 × (hidden_size × 704) × 4 bytes ≈ 30KB (at hidden=3584)
- Per MoE layer: 128 × 30KB ≈ 3.7MB
- All MoE layers: 30 × 3.7MB ≈ 112MB (just expert weights)
- Inference: only 8/128 experts compute per token = 6.25% active

---

## Implementation Plan

### Phase 1: MoE Weight Structs

#### Task 7.1: MoEExpertWeights and MoELayerWeights in model.mojo

**Files:** `src/mo/mogemma/model.mojo`

**Details:**
- `MoEExpertWeights` struct:
  - `gate_proj: TensorInfo` — [moe_intermediate_size, hidden_size]
  - `up_proj: TensorInfo` — [moe_intermediate_size, hidden_size]
  - `down_proj: TensorInfo` — [hidden_size, moe_intermediate_size]
- `MoELayerWeights` struct:
  - `router: TensorInfo` — [num_experts, hidden_size]
  - `experts: List[MoEExpertWeights]` — 128 experts
  - Standard attention weights (reuse from `LayerWeights`): q_proj, k_proj (=v_proj for K=V), o_proj
  - Norms: input_layernorm, post_attention_layernorm, pre/post_feedforward_layernorm
- `MoEModelWeights` struct:
  - `embed_tokens, norm, lm_head: TensorInfo`
  - `layers: List[MoELayerWeights]`
  - `get_embedding(token_id, out_ptr)` (same as ModelWeights)

**Tests:** Struct construction, expert count = 128, tensor shapes.

---

### Phase 2: Top-K Selection

#### Task 7.2: top_k op in ops.mojo

**Files:** `src/mo/mogemma/ops.mojo`

**Details:**
- `top_k(values_ptr, k: Int, size: Int, out_indices_ptr, out_values_ptr)`:
  - Partial sort: find k largest values and their indices from a vector of `size` elements
  - Output: `out_indices_ptr[0..k]` (indices), `out_values_ptr[0..k]` (values), both sorted descending
  - Algorithm: simple linear scan repeated k times (k=8 is tiny, size=128 is small — no need for heap-based partial sort)
- Used after softmax on router logits

**Tests:** Known vector top-8, ties, all-equal inputs, k=1 edge case.

---

### Phase 3: MoE Forward Functions

#### Task 7.3: forward_moe_router in layers.mojo

**Files:** `src/mo/mogemma/layers.mojo`

**Details:**
- `forward_moe_router(expert_indices_ptr, expert_weights_ptr, hidden_ptr, router_weight: TensorInfo, hidden_size: Int, num_experts: Int, top_k: Int, scratch_ptr)`:
  1. Router logits: `hidden @ router_weight^T` → [num_experts] via `vec_mat_mul`
  2. Softmax over all 128 logits
  3. Top-k selection → 8 indices + 8 weights
  4. Renormalize: selected weights sum to 1.0

**Tests:** Router output shape, weight sum = 1.0, indices in range [0, 127].

---

#### Task 7.4: forward_moe_experts in layers.mojo

**Files:** `src/mo/mogemma/layers.mojo`

**Details:**
- `forward_moe_experts(out_ptr, hidden_ptr, expert_indices: UnsafePointer[Int], expert_weights: UnsafePointer[Float32], experts: List[MoEExpertWeights], k: Int, hidden_size: Int, intermediate_size: Int, scratch_ptr)`:
  1. For each selected expert i (0..k-1):
     - `gate = experts[index].gate_proj @ hidden` → [intermediate_size]
     - `up = experts[index].up_proj @ hidden` → [intermediate_size]
     - `activated = geglu(gate, up)` → [intermediate_size]
     - `expert_out = experts[index].down_proj @ activated` → [hidden_size]
     - `out += expert_weights[i] * expert_out`
  2. Sparse: only 8 experts compute, other 120 skipped entirely

**Tests:** Single expert produces correct output, weighted sum of 8 experts matches manual computation, zero-weight expert contributes nothing.

---

#### Task 7.5: forward_moe_layer in layers.mojo

**Files:** `src/mo/mogemma/layers.mojo`

**Details:**
- `forward_moe_layer(out_ptr, x_ptr, weights: MoELayerWeights, layer_idx: Int, pos: Int, hidden_size: Int, num_heads: Int, num_kv_heads: Int, head_dim: Int, num_experts: Int, top_k: Int, intermediate_size: Int, kv_cache: KVCache, rope_tables: RoPETables, max_seq_len: Int, scratch_ptr)`:
  1. input_layernorm → attention (dispatches sliding/full, k_eq_v=True) → residual
  2. post_attention_layernorm
  3. pre_feedforward_layernorm → MoE block (router → top-8 experts → weighted sum) → post_feedforward_layernorm → residual
- K=V: pass `k_eq_v=True` to existing attention functions (already supported)
- 2 KV heads: pass `num_kv_heads=2` (GQA, already supported)

**Tests:** Layer produces correct output shape, attention uses K=V, MoE block activates 8 experts.

---

### Phase 4: Full Forward Pass

#### Task 7.6: forward_gemma4_moe_step in layers.mojo

**Files:** `src/mo/mogemma/layers.mojo`

**Details:**
- `forward_gemma4_moe_step(out_logits_ptr, token_id, pos, model: MoEModelWeights, hidden_size, num_heads, num_kv_heads, head_dim, num_experts, top_k, moe_intermediate_size, vocab_size, kv_cache, rope_tables, max_seq_len, scratch_ptr)`:
  1. Embed token → scale by `sqrt(hidden_size)`
  2. Loop 30 MoE layers
  3. Final RMSNorm
  4. LM head → logits

**Tests:** Full step with synthetic weights, correct logits shape.

---

### Phase 5: Core.mojo Wiring

#### Task 7.7: MoE weight building and step dispatch

**Files:** `src/mo/mogemma/core.mojo`

**Details:**
- `_build_moe_runtime(metadata, num_layers, num_experts)` — extracts:
  - Standard attention weights per layer
  - Router weights: `model.layers.{i}.block_sparse_moe.gate.weight`
  - 128 expert weight triplets per layer: `model.layers.{i}.block_sparse_moe.experts.{j}.w{1,2,3}.weight`
- Heap-allocate `MoEModelWeights`
- Flatten/hydrate with extended Appender/Hydrator (128 experts × 3 tensors per layer)
- `step_mojo` dispatches to `forward_gemma4_moe_step` when variant is MoE
- New architecture_overrides: `num_experts`, `moe_top_k`, `moe_intermediate_size`
- Scratch sizing: `hidden_size * 160 + max_seq_len * num_heads * 2 + num_experts * 4` (router buffer)

**Tests:** Init with MoE overrides, 128 experts per layer loaded, step dispatch.

---

#### Task 7.8: Python config wiring for 26B

**Files:** `src/py/mogemma/model.py`

**Details:**
- `_parse_gemma4_architecture` reads MoE keys from config.json:
  - `num_local_experts` → `num_experts`
  - `num_experts_per_tok` → `moe_top_k`
  - `moe_intermediate_size` → `moe_intermediate_size`
- Pass as architecture_overrides
- 26B uses vision (Ch5, 27-layer encoder), no audio
- `_detect_gemma4_variant` already handles this (checks `num_experts > 0`)

**Tests:** Config parsing for MoE params, verify no audio config for 26B.

---

### Phase 6: Integration & Verification

#### Task 7.9: End-to-end tests

**Files:** `src/py/tests/test_moe_config.py`, `src/mo/tests/test_moe.mojo`

**Details:**
- Router top-k with real-scale logits (128 experts)
- Expert dispatch correctness
- MoE layer forward shape verification
- Full step with synthetic 26B weights
- Config parsing for 26B variant
- Variant detection: MOE_26B_A4B when `num_experts > 0`
- No audio path for 26B

**Verification:**
```bash
CI=true uv run pytest -x -v
# Verify K=V is used for 26B:
grep -n "k_eq_v" src/mo/mogemma/layers.mojo
# Verify no audio in 26B path:
grep -r "audio" src/py/mogemma/model.py | grep -i "26b\|moe"
```

---

## Files Modified

| File | Changes |
|---|---|
| `src/mo/mogemma/model.mojo` | Add MoEExpertWeights, MoELayerWeights, MoEModelWeights |
| `src/mo/mogemma/layers.mojo` | Add forward_moe_router, forward_moe_experts, forward_moe_layer, forward_gemma4_moe_step |
| `src/mo/mogemma/ops.mojo` | Add top_k |
| `src/mo/mogemma/core.mojo` | MoE weight building, step dispatch, scratch sizing |
| `src/py/mogemma/model.py` | MoE config parsing |

## New Test Files

| File | Purpose |
|---|---|
| `src/py/tests/test_moe_config.py` | MoE config parsing, variant detection |
| `src/mo/tests/test_moe.mojo` | Router, expert dispatch, MoE layer, full step |

## Task Summary

| Task | Description | Status |
|---|---|---|
| 7.1 | MoE weight structs | [x] 379e65d |
| 7.2 | top_k op | [x] 379e65d |
| 7.3 | forward_moe_router | [x] 379e65d |
| 7.4 | forward_moe_experts | [x] 379e65d |
| 7.5 | forward_moe_layer | [x] 379e65d |
| 7.6 | forward_gemma4_moe_step | [x] 379e65d |
| 7.7 | Core.mojo MoE wiring | [x] 74fc44f |
| 7.8 | Python config wiring | [x] 379e65d |
| 7.9 | End-to-end tests | [x] 379e65d |

---

## Design Note: Expert Weight Loading Performance

Loading 11,520 MoE weight tensors (128 experts × 3 projections × 30 layers) from safetensors is a potential bottleneck. The current `SafetensorsLoader` does memory-mapped access, so no actual I/O happens until tensor data is read. The `_build_moe_runtime` function will iterate metadata keys matching `block_sparse_moe.experts.{j}` — this is O(n_tensors) string matching but happens once at init. No optimization needed for v1.
