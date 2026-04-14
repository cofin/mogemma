# Flow: moe-mojo-runtime

## Specification

### Goal

Make Gemma 4 26B-A4B-it run end-to-end on the Mojo backend. The Python
converter (sibling flow `orbax-safetensors-conversion` task 2.2a) already
emits a HF-style safetensors layout with the two-branch MoE contract; this
flow makes Mojo (a) load that contract via `core.mojo` hydration, (b) pack
it onto the GPU via `gpu_context.mojo`, and (c) execute the two-branch
forward pass with packed-expert kernels in `layers.mojo`.

### Parent PRD
- `gcs-orbax-migration` (unblocks live 26B inference over converted checkpoints).

### Depends On
- `orbax-safetensors-conversion` task 2.2a — Python `_iter_moe_transformer`
  is shipped (commit `8fdc954`). Tensor names emitted are the **contract**
  this flow consumes.

### Non-Goals

1. **Quantization.** All MoE weights are F32 at runtime. `*.weight_scale`
   tensors are ignored exactly as in the base-transformer path.
2. **Per-expert TensorInfo split.** The existing `MoEExpertWeights` list is
   retired; experts are stored packed (`[E, 2·I_moe, H]` and `[E, H, I_moe]`)
   because that's the layout the converter emits and the layout GPU packers
   prefer for a single contiguous allocation per-layer.
3. **Training / grad support.** Forward pass only.

### Safetensors Contract (from the Python converter)

**Global:**
- `model.embed_tokens.weight` `[V, H]`
- `lm_head.weight` `[V, H]` (tied copy)
- `model.norm.weight` `[H]`

**Per MoE layer N (30 total for 26B-A4B-it):**

| Safetensors name | Shape | Role |
|---|---|---|
| `model.layers.N.input_layernorm.weight` | `[H]` | RMSNorm before Q/K/V |
| `model.layers.N.post_attention_layernorm.weight` | `[H]` | RMSNorm after O projection |
| `model.layers.N.self_attn.q_proj.weight` | `[H_q, H]` | Q projection |
| `model.layers.N.self_attn.k_proj.weight` | `[H_kv, H]` | K projection |
| `model.layers.N.self_attn.v_proj.weight` | `[H_kv, H]` | V projection (== K for full-attn / KV-sharing) |
| `model.layers.N.self_attn.o_proj.weight` | `[H, H_q]` | O projection |
| `model.layers.N.self_attn.q_norm.weight` | `[head_dim]` | Q per-head RMSNorm |
| `model.layers.N.self_attn.k_norm.weight` | `[head_dim]` | K per-head RMSNorm |
| `model.layers.N.mlp.gate_proj.weight` | `[I_dense, H]` | Dense branch gate |
| `model.layers.N.mlp.up_proj.weight` | `[I_dense, H]` | Dense branch up |
| `model.layers.N.mlp.down_proj.weight` | `[H, I_dense]` | Dense branch down |
| `model.layers.N.moe_router.proj.weight` | `[E, H]` | Router logits matrix |
| `model.layers.N.moe_router.scale` | `[H]` | Router input scaling (RMSNorm-free input scale) |
| `model.layers.N.moe_router.per_expert_scale` | `[E]` | Per-expert weight scaling |
| `model.layers.N.moe_experts.gate_up_proj` | `[E, 2·I_moe, H]` | Packed per-expert gate+up weights |
| `model.layers.N.moe_experts.down_proj` | `[E, H, I_moe]` | Packed per-expert down weights |
| `model.layers.N.pre_feedforward_layernorm.weight` | `[H]` | Dense branch pre-norm |
| `model.layers.N.post_feedforward_layernorm_1.weight` | `[H]` | Dense branch post-norm (h1) |
| `model.layers.N.pre_feedforward_layernorm_2.weight` | `[H]` | MoE branch pre-norm |
| `model.layers.N.post_feedforward_layernorm_2.weight` | `[H]` | MoE branch post-norm (h2) |
| `model.layers.N.post_feedforward_layernorm.weight` | `[H]` | Extra norm (role unclear — hydrate, forward pass can ignore) |
| `model.layers.N.moe_skip_scale.weight` | `[1]` | Scalar; apply `residual * skip_scale` iff empirical value ≠ 0 |

### Reference forward pass (from HF `modeling_gemma4.py` + 26B inventory)

```
# Attention block (unchanged from base transformer wiring)
x1 = x + attn(input_layernorm(x))          # with post_attention_layernorm applied to attn_out per current impl

# Two-branch FFN
h1 = post_ffw1_norm(dense_mlp(pre_ffw_norm(x1)))
    dense_mlp(u) = down_proj(gelu(gate_proj(u)) * up_proj(u))

x_r      = pre_ffw2_norm(x1)
r        = rmsnorm_noscale(x1) * router_scale * hidden_size^-0.5
logits   = router_proj @ r                        # [E]
probs    = softmax(logits)
tw, ti   = topk(probs, k=num_experts_per_tok)    # both length-K
tw       = tw / tw.sum()                          # renormalize (HF convention)
tw       = tw * per_expert_scale[ti]              # per-expert scaling

h2_raw   = 0
for p in range(K):
    e          = ti[p]
    gate_up    = gate_up_proj[e] @ x_r            # [2·I_moe]
    gate, up   = gate_up[:I_moe], gate_up[I_moe:]
    expert_out = down_proj[e] @ (gelu(gate) * up) # [H]
    h2_raw    += tw[p] * expert_out
h2       = post_ffw2_norm(h2_raw)

out = x1 + h1 + h2                                # (skip_scale * x1 added iff skip_scale != 0)
```

---

## Implementation Plan

### Phase 0: Pinning + Diagnostic

- [ ] **0.1 Probe `skip_scale` on a real 26B checkpoint**
  - Run the Python converter over a real `google/gemma-4-26B-A4B-it` Orbax
    download, then read `model.layers.0.moe_skip_scale.weight` through
    `SafetensorsLoader`. Log the value per-layer; if every layer is ~0
    (< 1e-6), drop the `residual * skip_scale` term from the forward pass
    and skip hydration. If nonzero, keep it.
  - Record the observed value in `.agents/knowledge/gemma4-models.md` so
    we never re-guess. Update this spec's non-goals if skip_scale is
    empirically zero.

- [ ] **0.2 Freeze the `MoELayerWeights` struct layout**
  - Produce the final field list (no TensorInfo optional tricks — every
    field always emitted; forward pass branches on value, not presence).
  - Publish in this spec as the canonical layout before touching code.

### Phase 1: Mojo-side struct + hydration rewrite (no forward pass changes yet)

- [ ] **1.1 Rewrite `MoELayerWeights` in `model.mojo`**
  - Replace the current struct (`router`, `q/k/v/o_proj`, `q/k_norm`,
    4-norm set, `List[MoEExpertWeights]`) with the new 22-field struct:
    ```
    struct MoELayerWeights(Copyable, ImplicitlyCopyable, Movable):
        # Attention
        var q_proj, k_proj, v_proj, o_proj: TensorInfo
        var q_norm, k_norm: TensorInfo
        var input_layernorm, post_attention_layernorm: TensorInfo
        # Dense branch
        var pre_feedforward_layernorm: TensorInfo          # pre_ffw_norm
        var dense_gate_proj, dense_up_proj, dense_down_proj: TensorInfo
        var post_feedforward_layernorm_1: TensorInfo       # post_ffw1_norm
        # MoE branch
        var pre_feedforward_layernorm_2: TensorInfo        # pre_ffw2_norm
        var router_proj, router_scale, per_expert_scale: TensorInfo
        var expert_gate_up_proj, expert_down_proj: TensorInfo  # packed
        var post_feedforward_layernorm_2: TensorInfo       # post_ffw2_norm
        # Optional / investigatory
        var post_feedforward_layernorm: TensorInfo         # post_ffw_norm (may be zeroed)
        var moe_skip_scale: TensorInfo                     # (1,) scalar (may be zeroed per 0.1)
    ```
  - Delete `MoEExpertWeights`.
  - **Test-first:** `src/mo/tests/test_moe_weights.mojo` — construct a
    `MoELayerWeights` with synthetic tensor pointers, assert field count +
    read-back pointer equality.

- [ ] **1.2 Rewrite `_build_moe_from_runtime` in `core.mojo`**
  - Consume the new safetensors contract names. Each per-layer call opens
    the 22 tensors via `metadata_obj.get(...)`.
  - Missing optionals (`post_feedforward_layernorm`, `moe_skip_scale`)
    stay as null `TensorInfo(0,0,0)` when the safetensors shipped by
    older converters don't contain them.
  - **Test-first:** `src/mo/tests/test_moe_hydration.mojo` — synthesize a
    fake safetensors metadata dict with all 22 names, call the hydration
    path, assert every field is populated with the right pointer/shape.

- [ ] **1.3 Rewrite `_flatten_moe_weights` + `_hydrate_moe_weights` in `core.mojo`**
  - Update the flattening codec used to marshal MoE weight pointers across
    the Python ↔ Mojo FFI.
  - Pointer count per layer changes from (current ~22 including per-expert)
    to 22 fixed. Globals (embed_tokens, norm, lm_head) unchanged.
  - **Test-first:** round-trip a `MoEModelWeights` through flatten +
    hydrate; assert per-field pointer equality.

- [ ] **1.4 Update `gpu_context.mojo` packer**
  - `_pack_moe_layer` (or equivalent) now streams the 22 tensors into GPU
    buffers. Packed expert tensors are copied wholesale (no per-expert
    split); `gate_up_proj` is a single `[E * 2·I_moe * H]` allocation,
    `down_proj` is a single `[E * H * I_moe]` allocation.
  - **Test-first:** `src/mo/tests/test_moe_gpu_pack.mojo` — pack a tiny
    synthetic layer onto the CPU polyfill backend, assert device
    pointers round-trip to host identically.

### Phase 2: Forward pass

- [ ] **2.1 Port router to the new layout**
  - New `forward_moe_router` signature takes `router_proj` (matrix),
    `router_scale` (vector), `per_expert_scale` (vector) separately.
  - Implement the exact reference math:
    ```
    r = rmsnorm_noscale(x) * router_scale * hidden_size^-0.5
    logits = router_proj @ r
    probs = softmax(logits)
    tw, ti = topk(probs, K)
    tw /= tw.sum()
    tw *= per_expert_scale[ti]
    ```
  - **Test-first:** `test_moe_router_matches_reference` — compare Mojo
    output against a numpy reference implementation on a 16-expert, K=4
    fixture.

- [ ] **2.2 Packed-expert matmul kernel**
  - Replace the per-expert loop that reads individual `MoEExpertWeights`
    with a loop over the top-K selected indices that strides into the
    packed `[E, 2·I_moe, H]` and `[E, H, I_moe]` allocations.
  - Kernel invariants: one `_gemm_dispatch` per selected expert; no
    gather kernel yet — simple host-side loop over K is fine at K=8.
  - **Test-first:** `test_moe_experts_packed_matches_reference` — hydrate
    a 4-expert layer with random weights, run top-K=2, compare against
    numpy reference.

- [ ] **2.3 Two-branch forward pass**
  - Rewrite `forward_moe_layer` to compute `h1` (dense branch) and `h2`
    (MoE branch) in parallel and sum: `out = x1 + h1 + h2` (plus
    `x1 * skip_scale` iff Phase 0.1 says so).
  - Scratch-buffer layout must be recalculated — the current buffer plan
    assumes a single MLP branch.
  - **Test-first:** `test_forward_moe_layer_matches_reference` —
    end-to-end layer comparison against numpy on a 2-layer synthetic
    fixture with H=64, num_experts=4, K=2.

### Phase 3: Integration + verification

- [ ] **3.1 Wire the new path into `forward_pass` / model dispatch**
  - `core.mojo` model dispatch already branches on variant (Gemma4
    MoE); just update the tensor access to use the new field names.

- [ ] **3.2 Load a real 26B checkpoint end-to-end**
  - Run `mogemma.hub.HubManager().download_sync("google/gemma-4-26B-A4B-it")`
    — converter produces safetensors, Mojo hydrates, inference runs.
  - Greedy sample (temperature=0, top_k=1) on a 16-token prompt;
    compare against the HF-ref output for the same checkpoint.

- [ ] **3.3 Advisory benchmark**
  - Record tokens/sec on CPU + CPU-polyfill GPU for the 26B MoE path.
    Use `.agents/knowledge/performance.md` baselines as the target floor.

### Verification Gate
- [ ] `make test` green (Python + Mojo).
- [ ] `make lint` clean.
- [ ] Task 3.2 matches HF ref output byte-for-byte at temperature=0, top_k=1.
- [ ] skip_scale behavior decision from Phase 0.1 documented in
  `.agents/knowledge/gemma4-models.md`.

### Risks & Known Unknowns

1. **`post_feedforward_layernorm` role is undocumented.** HF reference
   doesn't use a 5th norm; the 26B checkpoint ships one anyway. Phase 0.1
   inspection decides: if all-zero / identity, we skip. If nonzero,
   figure out where it applies (candidates: on `h1 + h2` before residual
   add; on the final layer's output).
2. **skip_scale semantics unverified.** Same investigation path as above;
   if empirical value ≈ 0, it's dead weight we still have to hydrate
   (because the tensor exists in the safetensors file) but can ignore in
   the forward pass.
3. **Packed expert kernel perf at K=8.** A naive host-side loop over K
   dispatches K matmuls sequentially per layer. For 30 layers × K=8,
   that's 240 matmul calls per token. If this dominates wall time, we'll
   need a single gathered-matmul kernel — tracked as a follow-up.
4. **Scratch buffer sizing regression.** The current `forward_moe_layer`
   reserves scratch for one branch's intermediate; the new two-branch
   pass needs roughly 2x. If the caller pre-sizes scratch, we need to
   bump the allocation. Audit `core.mojo` scratch computation in Phase
   2.3.
