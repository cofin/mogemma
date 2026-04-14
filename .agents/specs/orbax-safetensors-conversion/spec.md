# Flow: orbax-safetensors-conversion

## Specification

### Goal
Convert downloaded Orbax/OCDBT checkpoints to HuggingFace-style safetensors files
with tensor names matching **exactly** what `src/mo/mogemma/core.mojo` loads via
`metadata_obj.get(...)`. Generate `config.json` from tensor shapes + known
per-variant architecture constants. Delete the Orbax artifacts after a verified
successful conversion so the cache ends up with a ready-to-use safetensors layout.

### Depends On
- Chapter 1 (`gcs-download-backend`) — download + OrbaxLoader are functional (✅ complete).

### Non-Goals (explicit scope cut from original spec)

1. **Audio encoder conversion is OUT of scope.** `core.mojo:1464` marks audio
   weight hydration as a TODO ("blocked on audio weight hydration — pending HF
   tensor name standardization"). There is no Mojo contract to satisfy yet.
   Produce audio tensors only if we later add `audio_tower.*` lookups in Mojo.
2. **Quantization (`*.weight_scale`) is OUT of scope.** `core.mojo:256-262` does
   call `get()` on scale tensors, but the returned value is treated as optional
   (used only when int8 weights are present). Converting all weights to F32
   matches current `SafetensorsLoader` behaviour. Scale emission can be added
   in a follow-up flow.
3. **Tensors Mojo does not load are skipped.** The original spec listed many
   "investigate/TBD" Orbax keys (`layer_N.mlp2.*`, `router_scale`,
   `per_expert_scale`, `per_layer_input_gate.w`, `pre_ffw2_norm`, etc.). None
   are referenced by `core.mojo`. Skip them — do not synthesize contract for
   hypothetical future use.

### Exact Mojo Tensor Contract (ground truth)

Extracted from `src/mo/mogemma/core.mojo` (verified line-by-line). These are
the ONLY names the converter must produce.

**Base transformer (always):**
- `model.embed_tokens.weight` (core.mojo:229)
- `model.norm.weight` (core.mojo:230)
- `lm_head.weight` (core.mojo:231) — tied to `embed_tokens` when Orbax lacks a separate tensor
- Per layer N in `[0, num_hidden_layers)`:
  - `model.layers.N.input_layernorm.weight` (:237)
  - `model.layers.N.post_attention_layernorm.weight` (:243)
  - `model.layers.N.self_attn.q_proj.weight` (:244)
  - `model.layers.N.self_attn.k_proj.weight` (:245)
  - `model.layers.N.self_attn.v_proj.weight` (:246)
  - `model.layers.N.self_attn.o_proj.weight` (:247)
  - `model.layers.N.mlp.gate_proj.weight` (:248)
  - `model.layers.N.mlp.up_proj.weight` (:249)
  - `model.layers.N.mlp.down_proj.weight` (:250)
  - `model.layers.N.self_attn.q_norm.weight` (:251)
  - `model.layers.N.self_attn.k_norm.weight` (:252)
  - `model.layers.N.pre_feedforward_layernorm.weight` (:253)
  - `model.layers.N.post_feedforward_layernorm.weight` (:254)

**PLE (E2B/E4B only), per layer N (core.mojo:447-462):**
- `model.layers.N.per_layer_input.per_layer_embedding.weight` (:452)
- `model.layers.N.per_layer_input.per_layer_projection.weight` (:458)
- `model.layers.N.per_layer_input.per_layer_norm.weight` (:460)

**Vision tower (all multimodal variants), per vision layer I (core.mojo:355-372):**
- `vision_tower.vision_model.embeddings.patch_embedding.weight` (:355)
- `vision_tower.vision_model.embeddings.position_embedding.weight` (:356)
- `vision_tower.vision_model.post_layernorm.weight` (:357)
- `multi_modal_projector.linear.weight` (:358)
- `vision_tower.vision_model.encoder.layers.I.self_attn.{q,k,v}_proj.weight` (:364-366)
- `vision_tower.vision_model.encoder.layers.I.self_attn.out_proj.weight` (:367)
- `vision_tower.vision_model.encoder.layers.I.mlp.fc1.weight` (:368)
- `vision_tower.vision_model.encoder.layers.I.mlp.fc2.weight` (:369)
- `vision_tower.vision_model.encoder.layers.I.layer_norm1.weight` (:370)
- `vision_tower.vision_model.encoder.layers.I.layer_norm2.weight` (:371)

**MoE (26B-A4B-it) — REVISED per 26B inventory + HF `transformers` `modeling_gemma4.py`:**

Gemma 4 MoE is a **two-branch parallel architecture**, NOT the HF
`block_sparse_moe` pattern originally assumed. Every one of the 30 layers has
BOTH a dense MLP (`mlp2`) and a routed MoE (`mlp`) branch; their outputs sum.

Reference forward pass (from HF modeling_gemma4.py):
```
# dense branch
h1 = post_ffw1_norm(dense_mlp(pre_ffw_norm(x)))
# MoE branch
x_r = pre_ffw2_norm(x)
_, topk_w, topk_idx = router(x)              # router consumes pre-norm x
h2_raw = experts(x_r, topk_idx, topk_w)
h2 = post_ffw2_norm(h2_raw)
out = residual + h1 + h2                     # plus skip_scale? — see below
```

Router internals:
```
r = RMSNorm(x, no_scale) * router_scale * hidden_size^-0.5
probs = softmax(router_logits @ r)
topk_w, topk_idx = topk(probs, k=num_experts_per_tok)
topk_w /= topk_w.sum(-1, keepdim=True)
topk_w *= per_expert_scale[topk_idx]
```

Expert internals (GEGLU per selected expert):
```
gate, up = (gate_up_proj[e] @ x_r).chunk(2, -1)
h = gelu(gate) * up
out_e = down_proj[e] @ h
final[token] += out_e * topk_w[token, pos]
```

**Per-layer Orbax tensors (all 30 layers):**
- Attention: `attn.q_einsum.w` `[16, 2816, 256]`, `attn.kv_einsum.w` or separate
  `attn.k_einsum.w` + `attn.v_einsum.w` (layer-dependent — local vs global attn),
  `attn.attn_vec_einsum.w` `[16, 256, 2816]`, `attn.query_norm.scale`,
  `attn.key_norm.scale`
- Dense branch: `mlp2.gating_einsum.w` `[2, 2112, 2816]` (split fc1/fc1_up),
  `mlp2.linear.w` `[2112, 2816]`
- MoE branch: `mlp.router_logits.w` `[2816, 128]`, `mlp.router_scale` `[2816]`,
  `mlp.per_expert_scale` `[128]`, `mlp.gating_einsum.w` `[128, 2, 704, 2816]`,
  `mlp.linear.w` `[128, 704, 2816]`
- Norms: `pre_attention_norm`, `post_attention_norm`, `pre_ffw_norm`,
  `post_ffw1_norm`, `pre_ffw2_norm`, `post_ffw2_norm`, `post_ffw_norm` (present
  alongside post_ffw1/post_ffw2; role unclear — emit faithfully, investigate in
  Task 2.2 verification).
- `skip_scale` `(1,)` per layer — scalar not in HF reference code. Emit as
  `...moe_skip_scale.weight`. Mojo forward may need `residual * skip_scale` if
  nonzero after inspection; confirm by checking a live checkpoint value (see
  Task 2.2).

### Orbax → Safetensors Mapping (confirmed transforms)

All shapes below assume BF16 source → F32 output. `H` = hidden_size,
`Hq` = head_dim × num_attention_heads (often = H), `Hkv` = head_dim × num_kv_heads.

| Orbax name | Safetensors name | Transform |
|---|---|---|
| `embedder.input_embedding` `[V, H]` | `model.embed_tokens.weight` `[V, H]` | none |
| `embedder.input_embedding` `[V, H]` | `lm_head.weight` `[V, H]` | none (tied copy) |
| `final_norm.scale` `[H]` | `model.norm.weight` `[H]` | none |
| `layer_N.pre_attention_norm.scale` | `...input_layernorm.weight` | none |
| `layer_N.post_attention_norm.scale` | `...post_attention_layernorm.weight` | none |
| `layer_N.pre_ffw_norm.scale` | `...pre_feedforward_layernorm.weight` | none |
| `layer_N.post_ffw_norm.scale` | `...post_feedforward_layernorm.weight` | none |
| `layer_N.attn.query_norm.scale` | `...self_attn.q_norm.weight` | none |
| `layer_N.attn.key_norm.scale` | `...self_attn.k_norm.weight` | none |
| `layer_N.attn.q_einsum.w` `[nq, H, d]` | `...q_proj.weight` `[nq·d, H]` | `w.transpose(0,2,1).reshape(-1, H)` |
| `layer_N.attn.kv_einsum.w` `[2, nkv, H, d]` | `...k_proj.weight` / `...v_proj.weight` | `w[0]`/`w[1]` then `.transpose(0,2,1).reshape(-1, H)` |
| `layer_N.attn.k_einsum.w` `[nkv, H, d]` | `...k_proj.weight` | (31B/MoE path — same reshape as q). Check for separate `v_einsum.w`. |
| `layer_N.attn.attn_vec_einsum.w` `[nq, d, H]` | `...o_proj.weight` `[H, nq·d]` | `w.reshape(-1, H).T` |
| `layer_N.mlp.gating_einsum.w` `[2, intermediate, H]` | `...mlp.gate_proj.weight` / `...mlp.up_proj.weight` | `w[0]` → gate, `w[1]` → up |
| `layer_N.mlp.linear.w` `[H, intermediate]` | `...mlp.down_proj.weight` `[H, intermediate]` | identity (verify — may need transpose) |
| **PLE** `embedder.per_layer_embeddings` `(V, L, H_ple)` e.g. `(262144, 35, 256)` | `model.layers.N.per_layer_input.per_layer_embedding.weight` `(V, H_ple)` | `arr[:, N, :]` (slice along middle axis) |
| **PLE** `layer_N.per_layer_projection.w` `(H_ple, H)` e.g. `(256, 1536)` | `...per_layer_projection.weight` `(H_ple, H)` | identity |
| **PLE** `layer_N.post_per_layer_input_norm.scale` `(H,)` e.g. `(1536,)` | `...per_layer_norm.weight` `(H,)` | identity |
| (PLE-adjacent Orbax tensors NOT consumed by Mojo — skipped) | — | `embedder.per_layer_model_projection.w`, `embedder.per_layer_projection_norm.scale`, `layer_N.per_layer_input_gate.w`, `layer_N.skip_scale` |
| **MoE router** `layer_N.mlp.router_logits.w` `[H, E]` | `...moe_router.proj.weight` `[E, H]` | `.T` |
| **MoE router** `layer_N.mlp.router_scale` `[H]` | `...moe_router.scale` `[H]` | identity |
| **MoE router** `layer_N.mlp.per_expert_scale` `[E]` | `...moe_router.per_expert_scale` `[E]` | identity |
| **MoE experts** `layer_N.mlp.gating_einsum.w` `[E, 2, I_moe, H]` | `...moe_experts.gate_up_proj` `[E, 2·I_moe, H]` | `arr.reshape(E, 2·I_moe, H)` (preserves `[gate; up]` order along axis-1) |
| **MoE experts** `layer_N.mlp.linear.w` `[E, I_moe, H]` | `...moe_experts.down_proj` `[E, H, I_moe]` | `arr.transpose(0, 2, 1)` |
| **MoE dense** `layer_N.mlp2.gating_einsum.w` `[2, I_dense, H]` | `...mlp.gate_proj.weight`, `...mlp.up_proj.weight` | `arr[0]` → gate, `arr[1]` → up |
| **MoE dense** `layer_N.mlp2.linear.w` `[I_dense, H]` | `...mlp.down_proj.weight` `[H, I_dense]` | `.T` |
| **MoE norms** `layer_N.pre_ffw_norm.scale` | `...pre_feedforward_layernorm.weight` | identity (dense pre-norm) |
| **MoE norms** `layer_N.post_ffw1_norm.scale` | `...post_feedforward_layernorm_1.weight` | identity (dense post-norm, h1) |
| **MoE norms** `layer_N.pre_ffw2_norm.scale` | `...pre_feedforward_layernorm_2.weight` | identity (MoE pre-norm) |
| **MoE norms** `layer_N.post_ffw2_norm.scale` | `...post_feedforward_layernorm_2.weight` | identity (MoE post-norm, h2) |
| **MoE norms** `layer_N.post_ffw_norm.scale` | `...post_feedforward_layernorm.weight` | identity (role unclear; emit) |
| **MoE extra** `layer_N.skip_scale` `(1,)` | `...moe_skip_scale.weight` `(1,)` | identity (investigate role) |
| **Vision** `vision_encoder.entry.input_projection.w` `[patch²·C, H_v]` | `...patch_embedding.weight` `[H_v, C, patch, patch]` | reshape + permute (HF convention) |
| **Vision** `vision_encoder.entry.pos_emb` `[tokens, H_v]` | `...position_embedding.weight` | none |
| **Vision** `ve.stacked_layers.block.*` `[L_v, ...]` | per-layer `encoder.layers.I.*` | split along leading axis (vmapped → per-layer) |
| **Vision** `embedder.mm_input_projection.w` | `multi_modal_projector.linear.weight` | likely `.T` — verify |

**Marked verification required:** entries labelled "verify" above, and **all
PLE shapes**, will be empirically confirmed against a live E2B-it checkpoint in
Task 1.0 before their transform code is written.

### config.json Generation

Orbax checkpoints downloaded from `gs://gemma-data` already include `config.json`
alongside the OCDBT artifacts (hub.py:362-363 downloads everything under the
`checkpoint_prefix`, which includes `config.json`). **Verify in Task 1.0** that
GCS ships config.json for every variant. If present — do nothing, keep it.

If a variant lacks config.json from GCS, generate one using:

- **Required keys** (validated by `hub.py:228-246`):
  - `model_type`: one of `gemma4`, `gemma4_text`, `gemma4_e2b`, `gemma4_e4b`,
    `gemma4_31b`, `gemma4_moe_26b` (startswith `gemma4`).
  - `num_hidden_layers`: inferred from `max(layer_N) + 1` in Orbax keys.

- **Architecture keys consumed by `model.py:109-203`**:
  - `sliding_window_size` or `sliding_window` (model.py:145)
  - `partial_rotary_factor` (:149)
  - `attention_k_eq_v` (:153)
  - `layer_types` (:158) — list like `["sliding", "full", ...]`
  - `vision_config.{num_hidden_layers, hidden_size, num_attention_heads, intermediate_size}` (:165-168)
  - `image_token_index` (:171)
  - `hidden_size_per_layer_input` (:176), `vocab_size_per_layer_input` (:179)
  - `use_double_wide_mlp` (:182)
  - `kv_sharing_layer_map` (:186)
  - `audio_token_index` (:191)
  - `num_local_experts` / `num_experts` (:196)
  - `num_experts_per_tok` (:199)
  - `moe_intermediate_size` (:200)

Per-variant constants come from `.agents/knowledge/gemma4-architecture.md`.

---

## Implementation Plan

### Phase 0: Empirical Verification (MUST run before code is written)

- [ ] **0.1 Download one E2B-it checkpoint and dump tensor inventory**
  - **Objective:** Freeze the exact Orbax tensor names and shapes for at least
    one PLE-bearing variant so transform code in Phase 1/2 is grounded.
  - **Targets:**
    - Cache path: `~/.cache/mogemma/google/gemma-4-e2b-it/`
    - Script location: `scripts/dump_orbax_inventory.py` (new, ephemeral — delete after Phase 1 closes)
  - **Implementation:**
    ```python
    from mogemma.hub import HubManager
    from mogemma.orbax_loader import OrbaxLoader
    path = HubManager().download_sync("google/gemma-4-e2b-it")
    # Use the streaming enumerator, NOT OrbaxLoader() which eager-loads:
    loader_stub = OrbaxLoader.__new__(OrbaxLoader)
    loader_stub.model_path = path
    names = loader_stub._enumerate_tensor_names()
    for n in names:
        arr = loader_stub._open_tensor(n)
        print(f"{n}\t{arr.shape}\t{arr.dtype}")
    ```
  - **Prerequisite:** GCS read access + ~5GB disk.
  - **Verification:** Captured output committed to
    `.agents/specs/orbax-safetensors-conversion/e2b-inventory.txt`. Use it to:
    1. Confirm PLE tensor layouts (Orbax-side shape vs `[L, ...]` stacked assumption).
    2. Confirm `config.json` is present in the downloaded checkpoint.
    3. Confirm `kv_einsum.w` shape is `[2, nkv, H, d]`.
  - **If discoveries contradict this spec, update the mapping table before Phase 1.**

- [ ] **0.2 Optionally dump 26B-A4B-it MoE inventory** (if disk + bandwidth allow)
  - Same procedure, confirms MoE expert packing `[E, 2, I, H]` and `[E, H, I]`.
  - If not feasible now, defer MoE transform code to a follow-up sub-flow; do NOT guess shapes.

### Phase 1: Streaming Converter Scaffolding + Base Transformer

- [ ] **1.1 Add streaming-friendly tensor access to `orbax_loader.py`**
  - **Objective:** Avoid loading full 150GB/81GB checkpoints into RAM. Current
    `OrbaxLoader.__init__` eager-loads every tensor (orbax_loader.py:65, 106-119).
  - **File:** `src/py/mogemma/orbax_loader.py`
  - **Add public staticmethods (no behavior change for existing API):**
    ```python
    @staticmethod
    def enumerate_tensors(model_path: Path) -> list[str]:
        """List tensor names without materializing any data."""
        stub = OrbaxLoader.__new__(OrbaxLoader)
        stub.model_path = Path(model_path)
        return stub._enumerate_tensor_names()

    @staticmethod
    def open_tensor(model_path: Path, name: str) -> np.ndarray:
        """Load a single tensor lazily; caller owns the returned array."""
        stub = OrbaxLoader.__new__(OrbaxLoader)
        stub.model_path = Path(model_path)
        arr = stub._open_tensor(name)
        if arr.dtype.name == "bfloat16":
            arr = arr.astype(np.float32)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return arr
    ```
  - **Test-first:** `src/py/tests/test_orbax_loader.py` — add
    `test_enumerate_tensors_does_not_materialize()` (mock tensorstore to assert
    `_open_tensor` is never called during enumeration) and
    `test_open_tensor_upcasts_bf16_to_f32()`.
  - **Verify:** `uv run pytest src/py/tests/test_orbax_loader.py -x`.

- [ ] **1.2 Create `src/py/mogemma/convert.py` module skeleton**
  - **Objective:** House the conversion logic. Streaming, name-mapping, and
    shard-writing live here.
  - **Public API:**
    ```python
    def convert_orbax_to_safetensors(
        model_path: Path,
        *,
        shard_size_bytes: int = 5 * 1024**3,
    ) -> list[Path]:
        """Convert the Orbax checkpoint at *model_path* in place.

        Writes either `model.safetensors` (single shard) or
        `model.safetensors.index.json` + `model-0000N-of-0000M.safetensors`
        (multi-shard) into *model_path*. Returns the list of written files.

        Raises ValueError if no Orbax artifacts are present.
        """
    ```
  - **Helpers (all private, module-level):**
    - `_variant_from_keys(keys: list[str]) -> Literal["base", "ple", "moe"]`
      — base if no `per_layer_*`/`router_*`, ple if `embedder.per_layer_*`
      present, moe if `router_logits` present.
    - `_layer_count(keys: list[str]) -> int` — `max` of `int(k.split("layer_")[1].split(".")[0])` + 1.
    - `_iter_base_transformer(path, keys, num_layers) -> Iterator[tuple[str, np.ndarray]]`
    - `_iter_ple(path, keys, num_layers) -> Iterator[tuple[str, np.ndarray]]`
    - `_iter_moe_experts(path, keys, num_layers, num_experts) -> Iterator[tuple[str, np.ndarray]]`
    - `_iter_vision(path, keys) -> Iterator[tuple[str, np.ndarray]]`
    - `_write_sharded(output_dir, tensor_iter, shard_size_bytes) -> list[Path]`
  - **Dependencies:** `numpy`, `safetensors.numpy.save_file`, `OrbaxLoader.enumerate_tensors`, `OrbaxLoader.open_tensor`.
  - **Test-first:** `src/py/tests/test_convert.py::test_variant_detection` with
    synthetic key lists for all 3 variants; `test_layer_count_handles_gaps`.

- [ ] **1.3 Implement base-transformer iterator**
  - Produces the 13 per-layer tensors + 3 globals listed in the contract.
  - Handle the `kv_einsum.w` (combined) vs `k_einsum.w` + separate `v_einsum.w`
    (31B/MoE) branching based on which key exists in the checkpoint.
  - **Test-first:** `test_base_transformer_mapping_produces_expected_names()` —
    build a tiny synthetic Orbax-shaped dict using mocked `open_tensor`, run
    `_iter_base_transformer`, assert the yielded name set equals the expected
    per-layer contract.
  - **Test:** `test_base_transformer_q_proj_shape_transform()` —
    input `[nq, H, d]` → output `[nq·d, H]`.
  - **Test:** `test_base_transformer_tied_lm_head()` — `lm_head.weight` is the
    same buffer as `embed_tokens.weight`.

- [ ] **1.4 Implement sharded safetensors writer**
  - **Policy:** buffer tensors in memory up to `shard_size_bytes`; when exceeded,
    flush to `model-0000{i}-of-0000{M}.safetensors`. Single-shard path writes
    `model.safetensors`. Multi-shard path also writes `model.safetensors.index.json`
    with `{"metadata": {"total_size": ...}, "weight_map": {name: shard_file}}`.
  - **M must be known before writing.** Two-pass strategy: first pass enumerates
    tensor byte counts, computes shard assignment; second pass opens each tensor
    and writes into its assigned shard.
  - **Test-first:** `test_writer_single_shard_under_threshold()`,
    `test_writer_multi_shard_produces_index_json()`,
    `test_writer_index_points_to_correct_shard()`.

### Phase 2: Variant-Specific Iterators

- [ ] **2.1 PLE iterator (blocked on 0.1 inventory)**
  - Transforms: per the (verified-in-0.1) mapping table.
  - Skips any non-Mojo-contract PLE Orbax keys.
  - **Test-first:** `test_ple_split_per_layer()` — synthetic `[V, L, H_ple]` → L outputs.

- [ ] **2.2 MoE iterator (26B-A4B-it) — two-branch architecture**
  - **Python iterator `_iter_moe_transformer(path, keys, num_layers)`** emits
    per layer:
    - Dense branch: `mlp.gate_proj.weight`, `mlp.up_proj.weight`,
      `mlp.down_proj.weight` (from `mlp2.gating_einsum.w` + `mlp2.linear.w`)
    - Router: `moe_router.proj.weight`, `moe_router.scale`,
      `moe_router.per_expert_scale`
    - Experts (packed, not per-expert split): `moe_experts.gate_up_proj`,
      `moe_experts.down_proj`
    - Norms: `pre_feedforward_layernorm`, `post_feedforward_layernorm_1`,
      `pre_feedforward_layernorm_2`, `post_feedforward_layernorm_2`,
      `post_feedforward_layernorm` (if present)
    - `moe_skip_scale.weight`
  - **Test-first tests** (in `src/py/tests/test_convert.py`):
    - `test_moe_dense_split()` — synthetic `[2, I_dense, H]` →
      `gate_proj.weight` + `up_proj.weight`; `[I_dense, H]` → `down_proj.weight.T`.
    - `test_moe_router_transpose()` — `[H, E]` → `[E, H]`.
    - `test_moe_experts_packed_reshape()` — `[E, 2, I_moe, H]` →
      `[E, 2·I_moe, H]` preserving gate-then-up interleave.
    - `test_moe_experts_down_transpose()` — `[E, I_moe, H]` → `[E, H, I_moe]`.
    - `test_moe_iterator_end_to_end()` — synthetic 2-layer Orbax fixture →
      verify all expected keys present + shapes match.
  - **Mojo-side changes** (REQUIRED for conversion to be meaningful):
    - `model.mojo`: replace existing `MoELayerWeights` with:
      ```
      struct MoELayerWeights:
          # Norms (5 required + 1 optional)
          pre_attention_norm, post_attention_norm: TensorInfo
          pre_ffw_norm, post_ffw1_norm: TensorInfo       # dense branch
          pre_ffw2_norm, post_ffw2_norm: TensorInfo      # MoE branch
          # Attention (same as base)
          q_proj, k_proj, v_proj, o_proj, q_norm, k_norm: TensorInfo
          # Dense MLP branch
          dense_gate_proj, dense_up_proj, dense_down_proj: TensorInfo
          # MoE branch
          router_proj, router_scale, per_expert_scale: TensorInfo
          expert_gate_up_proj, expert_down_proj: TensorInfo  # packed
          # Unique to Gemma 4 MoE
          skip_scale: TensorInfo                          # (1,) scalar
      ```
    - `layers.mojo`: rewrite `forward_moe_layer` to two-branch sum:
      ```
      pre_norm_d = rmsnorm(x, pre_ffw_norm)
      h1_raw = dense_mlp(pre_norm_d, dense_gate, dense_up, dense_down)
      h1 = rmsnorm(h1_raw, post_ffw1_norm)
      pre_norm_m = rmsnorm(x, pre_ffw2_norm)
      topk_w, topk_idx = router(x, router_proj, router_scale, per_expert_scale)
      h2_raw = moe_experts(pre_norm_m, topk_idx, topk_w, expert_gate_up, expert_down)
      h2 = rmsnorm(h2_raw, post_ffw2_norm)
      out = residual + h1 + h2     # add skip_scale·residual if inspection warrants
      ```
    - `core.mojo`: update `_build_moe_from_runtime`, `_flatten_moe_weights`,
      `_hydrate_moe_weights` for new tensor set.
    - `gpu_context.mojo`: update MoE packer to include all new tensors.
  - **Verification before finalizing:**
    - Print live `skip_scale` value from a layer — if ≈0, drop from forward;
      if nonzero, include `residual * skip_scale` in the sum.
    - Run synthetic conversion + Mojo load round-trip.

- [ ] **2.3 Vision iterator**
  - Patch embedding reshape + position embedding copy.
  - Stacked vision layers (`ve.stacked_layers.block.*` leading axis = vision layer) → per-layer split.
  - MLP `gating_einsum` → `fc1`; `linear` → `fc2` (with transpose).
  - **Test-first:** `test_vision_stacked_layers_split()`,
    `test_vision_patch_embedding_reshape_to_conv2d_layout()`.

### Phase 3: Hub Integration + Cleanup

- [ ] **3.1 Invoke conversion from `_finalize_download`**
  - **File:** `src/py/mogemma/hub.py`
  - **Insertion point:** between `staging_dir.rename(local_dir)` (line 327) and
    `return local_dir` (line 328). After rename, check:
    ```python
    if cls._has_orbax(local_dir) and not cls._has_safetensors(local_dir):
        from mogemma.convert import convert_orbax_to_safetensors
        convert_orbax_to_safetensors(local_dir)
        cls._cleanup_orbax_artifacts(local_dir)
    ```
  - **New static method:** `_cleanup_orbax_artifacts(path)` — removes
    `ocdbt.process_0/`, `manifest.ocdbt`, `_METADATA`, `_CHECKPOINT_METADATA`,
    `descriptor/`, `d/`, `commit_success.txt` (whichever exist). Keeps
    `config.json`, `tokenizer.model`, and the written `*.safetensors[.index.json]`.
  - **Atomicity:** cleanup only runs if `_has_safetensors` returns True after
    conversion. On failure, the Orbax artifacts stay so the user can retry.
  - **Test-first:** `test_finalize_download_converts_orbax()` — mock downloader
    to lay down a fake Orbax layout in the staging dir, patch
    `convert_orbax_to_safetensors` to write a dummy `model.safetensors`, assert
    Orbax files are deleted and safetensors survives.
  - **Test-first:** `test_finalize_download_preserves_orbax_on_conversion_failure()`
    — `convert_orbax_to_safetensors` raises → staging still contains Orbax.

- [ ] **3.2 Update `_has_model_files` priority (no-op if already correct)**
  - Current impl (hub.py:118-121) already checks safetensors first, then Orbax.
  - Action: verify no change needed; add a test that documents the precedence.
  - **Test-first:** `test_has_model_files_prefers_safetensors_over_orbax()`.

- [ ] **3.3 Wire `convert_orbax_to_safetensors` into both sync and async paths**
  - Both `download_sync` (hub.py:365) and `download_async` (hub.py:420) call
    `_finalize_download`, so a single change there covers both.
  - **Verification:** `grep -n _finalize_download src/py/mogemma/hub.py` — must
    only show 2 call sites + the definition.

### Phase 4: Integration Test + Manual Verification

- [ ] **4.1 Integration test: synthetic Orbax → SafetensorsLoader round trip**
  - **File:** `src/py/tests/test_convert.py`
  - **Setup:** tensorstore-free fake. Build a `FakeOrbaxPath` fixture that
    monkeypatches `OrbaxLoader.enumerate_tensors` / `OrbaxLoader.open_tensor`
    to return a curated dict simulating E2B base-transformer shapes.
  - **Assert:**
    1. `convert_orbax_to_safetensors(tmp_path)` writes `model.safetensors`.
    2. `SafetensorsLoader.can_load(tmp_path)` returns True.
    3. `SafetensorsLoader(tmp_path).get_tensor_metadata().keys()` is a superset
       of the base-transformer Mojo contract name set.
    4. `auto_loader(tmp_path)` returns a `SafetensorsLoader` instance.

- [ ] **4.2 Manual verification (optional, gated on GCS access)**
  - `uv run python -c "from mogemma.hub import HubManager; HubManager().download_sync('google/gemma-4-e2b-it')"`
  - Post-download, verify `~/.cache/mogemma/google/gemma-4-e2b-it/` contains
    ONLY `*.safetensors*`, `config.json`, `tokenizer.model` (no `ocdbt.*`).
  - Load the model end-to-end and run one inference step; assert it matches
    the existing Orbax-path output byte-for-byte at `temperature=0, top_k=1`.

### Verification Gate

- [ ] `uv run ruff check src/py/mogemma/convert.py src/py/mogemma/hub.py src/py/mogemma/orbax_loader.py` clean
- [ ] `uv run ruff format --check src/py/mogemma/` clean
- [ ] `uv run mypy src/py/mogemma/convert.py` clean
- [ ] `CI=true uv run pytest src/py/tests/test_convert.py src/py/tests/test_orbax_loader.py src/py/tests/test_hub.py` green
- [ ] `CI=true uv run pytest src/py/tests/ -q` overall suite green
- [ ] Task 4.2 manual verification passes (or explicitly deferred with reason in `learnings.md`)
- [ ] `learnings.md` updated with any PLE/MoE shape discoveries from Tasks 0.1/0.2

### Risks & Known Unknowns

1. **PLE tensor layouts unverified** — Task 0.1 mitigates; spec will be updated
   with actual shapes before Phase 2.1 code is written.
2. **MoE shape guesses unverified** — Task 0.2 mitigates; if skipped, 2.2
   raises NotImplementedError rather than silently producing wrong outputs.
3. **Two-pass sharding re-opens every tensor** — roughly doubles I/O for large
   models. Acceptable given write happens once per download. If profiling
   shows this as a bottleneck, switch to "single-pass with estimated total" +
   post-rename to renumber shards.
4. **Audio users** — any caller expecting audio safetensors will observe the
   same failure as today (Mojo doesn't load them). No regression.
