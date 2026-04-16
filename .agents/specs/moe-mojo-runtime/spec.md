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

combined = post_ffw_norm(h1 + h2)                 # H-A placement (HF-ref confirmed)
out      = x1 + combined
out     *= layer_scalar                           # HF-ref confirmed multiplicative output scalar
```

### Reference precedent for the unknowns — CLOSED 2026-04-16

Both "unknowns" (`post_feedforward_layernorm.weight` shape `[H]` and
`moe_skip_scale.weight` shape `[1]`) are resolved definitively by
HuggingFace `transformers` `src/transformers/models/gemma4/modeling_gemma4.py`
`Gemma4TextDecoderLayer.forward()`. See Phase 0.1c for the exact
quoted source. Summary of the resolution:

- `post_feedforward_layernorm` → applied to `(h1 + h2)` before
  residual add (hypothesis H-A). Confirmed by direct source reading,
  not inference.
- `moe_skip_scale` → HF's `layer_scalar` buffer; applied as a
  **multiplicative scalar on the entire layer output**, last
  operation before return. NOT added to the residual.

The hypothesis enumeration framework below is preserved for
historical record of the decision process; all other hypotheses
(H-B, H-C, H-D) are formally ruled out.

**`post_feedforward_layernorm` — precedent exists.**
The name matches the Gemma 2 / Gemma 3 `post_feedforward_layernorm`
introduced in `google/gemma_pytorch`. In those models the semantics are
unambiguous (verified against
`gemma_pytorch/gemma/model.py::Gemma2DecoderLayer.forward`):

```python
residual = hidden_states
if self.pre_feedforward_layernorm is not None:
    hidden_states = self.pre_feedforward_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
if self.post_feedforward_layernorm is not None:
    hidden_states = self.post_feedforward_layernorm(hidden_states)
hidden_states = residual + hidden_states
```

i.e. `out = residual + post_ffw_norm(mlp(pre_ffw_norm(residual)))`.
`use_post_ffw_norm=True` is set for every Gemma 2 and Gemma 3 config in
`gemma_pytorch/gemma/config.py` (2b-v2, 9b, 27b, 1b, 4b, 12b, 27b-v3).

The 26B-A4B-it Orbax checkpoint ships **all three** norms per layer
(`post_ffw1_norm`, `post_ffw2_norm`, `post_ffw_norm`), all at shape
`(H,) = (2816,)`. Combined with the Gemma 2/3 precedent and the
two-branch parallel FFN topology, the four remaining hypotheses are:

| ID | Placement | Forward-pass formula | Notes |
|---|---|---|---|
| H-A | Combined-branch post-norm before residual | `out = x1 + post_ffw_norm(post_ffw1_norm(h1_raw) + post_ffw2_norm(h2_raw))` | Preserves Gemma 2/3 pattern at block level; per-branch norms are additional. **Leading hypothesis.** |
| H-B | Single global norm replacing per-branch | `out = x1 + post_ffw_norm(h1_raw + h2_raw)` | Per-branch norms are vestigial/unused. Unlikely (they're non-identity in inventory). |
| H-C | Vestigial tensor | ignore — `out = x1 + post_ffw1_norm(h1_raw) + post_ffw2_norm(h2_raw)` | Checkpoint carries the Gemma 3 name for compatibility but Gemma 4 MoE uses per-branch norms only. |
| H-D | Block output norm (post-residual) | `out = post_ffw_norm(x1 + h1 + h2)` | Unusual; no precedent. Last-resort fallback. |

Decision is made by Phase 0.1c (empirical enumeration against HF-ref
logits on a one-token prompt).

**`moe_skip_scale` — no precedent.**
No public Gemma / DeepMind reference documents this tensor. Searches
across `google-deepmind/gemma`, `google/gemma_pytorch`, Flax/Flaxformer,
and third-party Gemma 4 write-ups (kaitchup, Grootendorst, HF blog,
MindStudio, Modular, DeepMind blog) turn up **zero hits for
`skip_scale` on any MoE variant**. It's a genuinely novel Gemma 4
addition. The only observable data we have is the tensor shape `(1,)`
and dtype `float32` from the 26B-A4B-it Orbax inventory. Decision is
made purely from empirical value distribution per Phase 0.1a.

---

## Implementation Plan

### Phase 0: Empirical Probe + Pinning

**Goal of this phase:** close the two research unknowns (`moe_skip_scale`
and `post_feedforward_layernorm`) against a real 26B-A4B-it checkpoint
*before* wiring any forward-pass changes. Output of this phase is a
decision in `.agents/knowledge/gemma4-models.md` that pins the forward
path deterministically.

#### Prerequisites (user-run)

The probe requires the 26B-A4B-it Orbax checkpoint (~52 GB). The
`gs://gemma-data` bucket is **public, anonymous-read** — the project's
`HubManager` constructs its `GCSStore` with `skip_signature=true` (see
`src/py/mogemma/hub.py:42-52`), so no `gcloud auth`, service account,
or ADC setup is required. A bare `uv run` on a machine with network
access is sufficient.

```bash
# 1. Hydrate the checkpoint via the project's existing HubManager.
#    Downloads to MOGEMMA_CACHE_DIR (defaults to ~/.cache/mogemma).
#    Anonymous read — no gcloud auth required.
uv run python -c "
from mogemma.hub import HubManager
path = HubManager().download_sync('google/gemma-4-26B-A4B-it')
print(f'Checkpoint at: {path}')
"

# 2. Confirm the Orbax keys include skip_scale + post_ffw_norm.
uv run python -c "
from mogemma.convert import OrbaxLoader
from pathlib import Path
root = Path.home() / '.cache' / 'mogemma' / 'google' / 'gemma-4-26B-A4B-it'
keys = list(OrbaxLoader.enumerate_tensors(root))
print('skip_scale keys:',    sum(1 for k in keys if k.endswith('.skip_scale')))
print('post_ffw_norm keys:', sum(1 for k in keys if k.endswith('.post_ffw_norm.scale')))
print('expected: 30 of each for 26B-A4B-it')
"
```

If `HubManager.download_sync` stalls, the raw obstore fallback uses
the same anonymous config:

```python
from obstore.store import GCSStore
store = GCSStore("gemma-data", config={"skip_signature": "true"})
# Streaming per-file download; see src/py/mogemma/hub.py::_download_file.
```

- [x] **0.1a Probe `moe_skip_scale` distribution** (2026-04-16)
  - **Result: LIVE.** All 30 layers have positive values, min=+0.0705,
    max=+0.8151, mean=+0.6244, std=0.197. No layer is near zero. Full
    per-layer data in `.agents/knowledge/gemma4-26b-skip-scale.csv`.
    Decision documented in `.agents/knowledge/gemma4-models.md`.
    Phase 4.1 Branch B is pinned.
  - **Runner:** `scripts/probe_moe_unknowns.py` (created in this task —
    lives outside `src/` so it isn't shipped; run via
    `uv run python scripts/probe_moe_unknowns.py skip_scale`).
  - **What it does:**
    - Enumerates `layer_N.skip_scale` for `N = 0..29` using
      `OrbaxLoader.open_tensor`.
    - Prints: per-layer value, min/max/mean across layers, count of
      values with `abs(v) < 1e-6`.
    - Writes `.agents/knowledge/gemma4-26b-skip-scale.csv` with columns
      `layer,value,dtype`.
  - **Decision gate:**
    - `max(|v|) < 1e-6` across all 30 layers → **skip_scale is dead.**
      Phase 4 wires the forward pass without any `skip_scale` term.
      Hydration still emits the tensor for checkpoint compatibility but
      the forward pass never reads it.
    - `max(|v|) >= 1e-6` → **skip_scale is live.** Phase 4 wires
      `out += x1 * skip_scale[0]` into the residual sum. Record whether
      the sign is positive (residual amplification) or negative
      (residual damping) for Phase 3 parity diagnostics.
  - **Artifact:** append findings to
    `.agents/knowledge/gemma4-models.md` under a new
    `### MoE layer — skip_scale (26B-A4B-it)` heading with the CSV
    summary and decision.

- [x] **0.1b Probe `post_ffw_norm` distribution** (2026-04-16)
  - **Result: LIVE, highly non-identity.** `||w-1||_inf` ranges
    0.97 (layer 11) → 21.95 (layer 0); layer-mean ranges 0.87 → 13.75
    with characteristic boundary-layer amplification (layers 0, 29
    strongest). Cross-check: `post_ffw1_norm` and `post_ffw2_norm`
    are also live (`||w-1||_inf` > 88 on layer 0), which rules out
    H-B (single-norm-replacing-per-branch). H-C (vestigial) is ruled
    out by the learned training dynamic. **Remaining hypotheses: H-A
    vs. H-D.** Full data in
    `.agents/knowledge/gemma4-26b-post-ffw-norm.csv`. Decision
    documented in `.agents/knowledge/gemma4-models.md`.
  - **Runner:** same `scripts/probe_moe_unknowns.py post_ffw_norm`.
  - **What it does:**
    - Enumerates `layer_N.post_ffw_norm.scale` for `N = 0..29`.
    - For each layer computes: mean, stddev, `||w − 1||_inf` (distance
      from identity), and the 10-quantile histogram.
    - Writes
      `.agents/knowledge/gemma4-26b-post-ffw-norm.csv` (layer, mean,
      std, inf_dist_from_one, quantile_00, …, quantile_10).
  - **Decision gate:**
    - `max(||w − 1||_inf) < 1e-3` across layers → **norm is identity.**
      Phase 4 treats H-C as confirmed (vestigial tensor) — per-branch
      norms only.
    - `max(||w − 1||_inf) >= 1e-3` → **norm is live.** Proceed to
      Phase 0.1c to distinguish H-A vs. H-B vs. H-D.
  - **Artifact:** same knowledge file, new
    `### MoE layer — post_feedforward_layernorm (26B-A4B-it)` heading.

- [x] **0.1c Placement hypothesis CLOSED by HF reference source** (2026-04-16)
  - **No probe harness needed.** HuggingFace `transformers`
    `src/transformers/models/gemma4/modeling_gemma4.py` ships the full
    reference implementation of `Gemma4TextDecoderLayer`. Fetched and
    verified 2026-04-16.
  - **Result: H-A confirmed** — `post_feedforward_layernorm` applies
    to `(h1 + h2)` before the residual add. Quoted HF forward:

    ```python
    # In Gemma4TextDecoderLayer.forward():
    residual = hidden_states
    hidden_states = self.pre_feedforward_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)

    if self.enable_moe_block:
        hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

        hidden_states_flat = residual.reshape(-1, residual.shape[-1])
        _, top_k_weights, top_k_index = self.router(hidden_states_flat)
        hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
        hidden_states_2 = self.experts(
            hidden_states_2, top_k_index, top_k_weights
        )
        hidden_states_2 = hidden_states_2.reshape(residual.shape)
        hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

        hidden_states = hidden_states_1 + hidden_states_2

    hidden_states = self.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    # ... optional PLE gate residual ...

    hidden_states *= self.layer_scalar
    return hidden_states
    ```

  - **Correction to Phase 0.1a decision:** the Orbax `layer_N.skip_scale`
    tensor maps to HF's `layer_scalar` (buffer of shape `(1,)` on
    `Gemma4TextDecoderLayer`, state-dict key
    `layers.{N}.layer_scalar`). **It is NOT added to the residual.**
    It is a **multiplicative scalar on the entire layer output**
    applied as the very last operation before returning:

    ```
    out = layer_scalar * (residual + post_ffw_norm(h1 + h2) + optional_ple_branch)
    ```

  - **Additional nuances uncovered (critical for correctness):**
    1. **Dense and MoE branches use different pre-norms on different
       inputs.** Dense uses `pre_feedforward_layernorm` (the
       un-numbered one) applied to the pre-attention residual;
       MoE uses `pre_feedforward_layernorm_2` applied to the **raw
       flattened residual** (`residual.reshape(-1, H)`), **not** to
       the dense-branch pre-normed tensor. The spec's pseudocode
       earlier in this file correctly describes `pre_ffw2_norm(x1)`
       — `x1` there is the raw residual (post-attention,
       pre-dense-pre-norm), matching HF.
    2. **Router input is the raw flattened residual**, not a
       normed value. The router internally applies its own
       `rmsnorm_noscale` to its input per the spec's router math.
    3. **Both `hidden_states_1` (dense) and `hidden_states_2` (MoE)
       are summed *before* `post_feedforward_layernorm`**, not
       after. H-A placement confirmed definitively.

  - **Canonical forward formula (replaces the earlier pseudocode):**

    ```text
    x1 = x + attn(input_layernorm(x))         # attention block unchanged

    # Dense branch
    dense_in = pre_feedforward_layernorm(x1)
    h1_raw   = dense_mlp(dense_in)
    h1       = post_feedforward_layernorm_1(h1_raw)

    # MoE branch (operates on RAW x1 flattened, not dense_in)
    x_flat    = x1.reshape(-1, H)
    _, tw, ti = router(x_flat)
    moe_in    = pre_feedforward_layernorm_2(x_flat)
    h2_raw    = experts(moe_in, ti, tw).reshape(x1.shape)
    h2        = post_feedforward_layernorm_2(h2_raw)

    # Combine + final layer scalar
    combined  = h1 + h2
    combined  = post_feedforward_layernorm(combined)
    out       = x1 + combined
    # (optional PLE residual branch applies here)
    out       = out * layer_scalar
    ```

  - **Artifact:** `.agents/knowledge/gemma4-26b-post-ffw-norm-placement.md`
    records this closure with the HF source pointer.
  - **Phase 0.1c runner/probe script is no longer needed.** The probe
    script spec for `placement` subcommand in 0.1c is retired.

- [x] **0.2 Freeze the `MoELayerWeights` struct layout** [8058ecb]
  - Produce the final field list (no TensorInfo optional tricks — every
    field always emitted; forward pass branches on value, not presence).
  - Publish in this spec as the canonical layout before touching code.

### Phase 1: Mojo-side struct + hydration rewrite (no forward pass changes yet)

- [x] **1.1 Rewrite `MoELayerWeights` in `model.mojo`** [8058ecb]
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

- [x] **1.2 Rewrite `_build_moe_from_runtime` in `core.mojo`** [f7fae54]
  - Consume the new safetensors contract names. Each per-layer call opens
    the 22 tensors via `metadata_obj.get(...)`.
  - Missing optionals (`post_feedforward_layernorm`, `moe_skip_scale`)
    stay as null `TensorInfo(0,0,0)` when the safetensors shipped by
    older converters don't contain them.
  - **Test-first:** `src/mo/tests/test_moe_hydration.mojo` — synthesize a
    fake safetensors metadata dict with all 22 names, call the hydration
    path, assert every field is populated with the right pointer/shape.

- [x] **1.3 Rewrite `_flatten_moe_weights` + `_hydrate_moe_weights` in `core.mojo`** [8058ecb]
  - Update the flattening codec used to marshal MoE weight pointers across
    the Python ↔ Mojo FFI.
  - Pointer count per layer changes from (current ~22 including per-expert)
    to 22 fixed. Globals (embed_tokens, norm, lm_head) unchanged.
  - **Test-first:** round-trip a `MoEModelWeights` through flatten +
    hydrate; assert per-field pointer equality.

- [!] **1.4 Update `gpu_context.mojo` packer**
  - `_pack_moe_layer` (or equivalent) now streams the 22 tensors into GPU
    buffers. Packed expert tensors are copied wholesale (no per-expert
    split); `gate_up_proj` is a single `[E * 2·I_moe * H]` allocation,
    `down_proj` is a single `[E * H * I_moe]` allocation.
  - **Test-first:** `src/mo/tests/test_moe_gpu_pack.mojo` — pack a tiny
    synthetic layer onto the CPU polyfill backend, assert device
    pointers round-trip to host identically.

### Phase 2: Forward pass

- [x] **2.1 Port router to the new layout** [8058ecb]
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

- [x] **2.2 Packed-expert matmul kernel** [8058ecb]
  - Replace the per-expert loop that reads individual `MoEExpertWeights`
    with a loop over the top-K selected indices that strides into the
    packed `[E, 2·I_moe, H]` and `[E, H, I_moe]` allocations.
  - Kernel invariants: one `_gemm_dispatch` per selected expert; no
    gather kernel yet — simple host-side loop over K is fine at K=8.
  - **Test-first:** `test_moe_experts_packed_matches_reference` — hydrate
    a 4-expert layer with random weights, run top-K=2, compare against
    numpy reference.

- [x] **2.3 Two-branch forward pass** [f7fae54]
  - Rewrite `forward_moe_layer` to compute `h1` (dense branch) and `h2`
    (MoE branch) in parallel and sum: `out = x1 + h1 + h2` (plus
    `x1 * skip_scale` iff Phase 0.1 says so).
  - Scratch-buffer layout must be recalculated — the current buffer plan
    assumes a single MLP branch.
  - **Test-first:** `test_forward_moe_layer_matches_reference` —
    end-to-end layer comparison against numpy on a 2-layer synthetic
    fixture with H=64, num_experts=4, K=2.

### Phase 3: Integration + verification

- [x] **3.1 Wire the new path into `forward_pass` / model dispatch** [f7fae54]
  - `core.mojo` model dispatch already branches on variant (Gemma4
    MoE); just update the tensor access to use the new field names.

- [!] **3.2 Load a real 26B checkpoint end-to-end with dual parity gates**
  - **Setup:** Phase 0 must be complete — `skip_scale` decision and
    `post_ffw_norm` hypothesis are pinned in
    `.agents/knowledge/gemma4-models.md`.
  - **Conversion + hydration:**
    `mogemma.hub.HubManager().download_sync("google/gemma-4-26B-A4B-it")`
    produces safetensors; Mojo hydrates via the Phase 1 path.
  - **Gate A — layerwise cosine-similarity diagnostic (non-gating):**
    Capture hidden states after layers `[0, 1, 5, 15, 29]` and after
    the final RMSNorm from both (a) Mojo forward pass and (b) HF
    `AutoModelForCausalLM` reference, on a fixed 8-token prompt
    `"The quick brown fox jumps over the lazy dog"` tokenized by the
    shared Gemma 4 tokenizer. Compute `1 − cosine(mojo_h, hf_h)` per
    recorded layer. Expect `< 1e-6` for all layers; any layer `>= 1e-5`
    indicates a numerical regression — investigate before Gate B.
    Artifact: `.agents/knowledge/gemma4-26b-parity.md`.
  - **Gate B — greedy bit-exact parity (gating):**
    Run greedy sampling (`temperature=0`, `top_k=1`) for a 32-token
    continuation from the same prompt. Token IDs from Mojo MUST match
    HF reference bit-exactly. Any divergence = failure.
  - **Failure-diagnostic protocol:** if Gate B fails but Gate A is
    clean up to some layer `L`, divergence is introduced after `L`.
    Binary-search layer-by-layer cosine to localize; the first layer
    with cosine drift above 1e-5 is the bug site. Most likely culprits
    (in priority order): expert-routing top-K selection (float
    tie-breaking), packed-expert matmul index stride, post_ffw_norm
    placement (re-run Phase 0.1c on that specific layer).

- [!] **3.3 Advisory benchmark**
  - Record tokens/sec on CPU + CPU-polyfill GPU for the 26B MoE path.
    Use `.agents/knowledge/performance.md` baselines as the target floor.
  - Profile per-layer breakdown: attention, dense MLP, router,
    packed-expert matmul, norms. Flag any component `> 40%` of
    per-layer time as a candidate for the gathered-matmul follow-up.

### Phase 4: Post-probe forward-pass wiring (gated on Phase 0 outcomes)

This phase only runs after Phase 0 is complete. Each task selects one
branch based on the knowledge-file decisions.

- [!] **4.1 Wire `layer_scalar` (Orbax `skip_scale`) as multiplicative output scalar**
  - **Confirmed placement:** `out *= weights.moe_skip_scale.ptr[0]`
    applied as the **last operation in `forward_moe_layer`**, after
    the residual add and after any optional PLE gate residual.
    **Do NOT add it to the residual sum.** Reference: HF
    `Gemma4TextDecoderLayer.forward()` — `hidden_states *=
    self.layer_scalar; return hidden_states`.
  - **Unit test** (`src/mo/tests/test_moe_layer_scalar.mojo`):
    construct a 2-layer synthetic fixture with `layer_scalar =
    {1.0, 0.5}`; verify layer-0 output is unchanged vs. a
    no-scalar reference, layer-1 output is exactly half the
    no-scalar reference.
  - **Naming note:** Orbax ships the tensor as `layer_N.skip_scale`;
    the project converter maps it to `model.layers.N.moe_skip_scale.weight`.
    HF reference calls it `layer_scalar`. All three names refer to
    the same `(1,)` buffer. The Mojo struct field stays
    `moe_skip_scale` for consistency with the safetensors contract.

- [!] **4.2 Wire `post_feedforward_layernorm` (H-A confirmed)**
  - **Pinned formula:** see the "Canonical forward formula" quoted
    in Phase 0.1c. Specifically:

    ```text
    h1 = post_ffw1_norm(dense_mlp(pre_ffw_norm(x1)))
    h2 = post_ffw2_norm(experts(pre_ffw2_norm(x1.reshape(-1, H)),
                                router_top_k(x1.reshape(-1, H))))
                     .reshape(x1.shape)
    out = x1 + post_ffw_norm(h1 + h2)
    ```

    Note: `pre_ffw2_norm` is applied to the **raw flattened
    residual**, NOT to `pre_ffw_norm(x1)`. This differs from the
    original spec pseudocode that used `pre_norm_m = pre_ffw2_norm(x1)`
    without explicit reshape — the reshape is a no-op for rank-2
    inputs but critical for the router's expected `(batch*seq, H)`
    layout.
  - **Scratch-buffer impact:** `forward_moe_layer` now needs space
    for `h1`, `h2`, and a third `[B*S, H]`-sized buffer for
    `(h1 + h2)` before `post_ffw_norm`. Audit
    `core.mojo::_compute_moe_scratch` (Phase 2.3 callsite) and
    bump allocation by one `[B*S, H]` tile.
  - **Test** (`src/mo/tests/test_moe_post_ffw_norm_placement.mojo`):
    numpy reference implementing the canonical formula (ported
    inline from the HF quote), single-layer forward with
    deterministic-seed random weights, compare Mojo output to
    numpy reference with `||diff||_inf < 1e-5` gate.

- [!] **4.3 Refresh 26B-A4B-it parity after Phase 4.1 + 4.2**
  - Re-run Phase 3.2 Gate B. Must pass before calling the flow done.

### Verification Gate

- [x] `make test` green (Python + Mojo). [f7fae54]
- [x] `make lint` clean. [f7fae54]
- [!] Phase 0.1a result recorded in `.agents/knowledge/gemma4-models.md`.
- [!] Phase 0.1b result recorded in `.agents/knowledge/gemma4-models.md`.
- [!] Phase 0.1c hypothesis pinned in
  `.agents/knowledge/gemma4-26b-post-ffw-norm-placement.md` (if 0.1b
  showed non-identity).
- [!] Phase 3.2 Gate A cosine drift `< 1e-5` at all probed layers.
- [!] Phase 3.2 Gate B greedy bit-exact parity with HF reference
  (32-token continuation from the fixed 8-token prompt).

### Risks & Known Unknowns

1. **`post_feedforward_layernorm` placement uncertainty is now bounded
   to four enumerable hypotheses (H-A..H-D) per the "Reference
   precedent" section.** Phase 0.1c decides deterministically. Risk:
   none match, which halts the flow and forces a research sub-flow
   on undocumented forward-pass terms.
2. **`skip_scale` has no public precedent.** The probe in 0.1a is the
   only ground truth; if the value is nonzero but tiny (`< 1e-4`), the
   forward-pass effect may be numerically invisible — Gate A cosine
   drift will tell us whether to include or omit the term.
3. **Packed expert kernel perf at K=8.** A naive host-side loop over K
   dispatches K matmuls sequentially per layer. For 30 layers × K=8,
   that's 240 matmul calls per token. If this dominates wall time, we'll
   need a single gathered-matmul kernel — tracked as a follow-up.
4. **Scratch buffer sizing regression.** The current `forward_moe_layer`
   reserves scratch for one branch's intermediate; the two-branch
   pass needs roughly 2x, and H-A adds one more `[H]`-sized buffer.
   Audit `core.mojo` scratch computation in Phase 4.2.
5. **Checkpoint availability.** Phase 0 requires a local 26B-A4B-it
   checkpoint (~52 GB). `gs://gemma-data` is public and anonymous-read
   (`skip_signature=true`), so the only failure mode is transient
   network loss — retry policy lives in `HubManager._download_file`.
