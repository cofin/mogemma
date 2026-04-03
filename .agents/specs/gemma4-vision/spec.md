# Chapter 5: Multimodal Vision Pipeline

**Flow ID:** `gemma4-vision`
**Parent PRD:** `gemma4-rewrite`
**Beads Epic:** `mogemma-kuil.5`
**Depends on:** Chapter 4 (working text inference at `577942c`)
**Status:** Planned

---

## Goal

Add image and video understanding to the Gemma 4 runtime. The vision encoder is a SigLIP-based ViT shared by all four variants — only the layer count differs: 27 layers for 31B/26B, 16 layers for E2B/E4B. This chapter builds the full pipeline from raw pixels to vision tokens injected into the decoder's text sequence.

## Current Codebase State (Post-Ch4)

**What exists:**
- `model.mojo`: `TensorInfo`, `LayerWeights`, `ModelWeights`, `KVCache`, `RoPETables` — text-only. No vision structs.
- `layers.mojo`: `forward_sliding_attention`, `forward_full_attention`, `forward_gemma4_layer`, `forward_gemma4_step`, `forward_mlp`, `_gemm_dispatch` — all single-token decoder operations.
- `ops.mojo`: `geglu`, `rope_rotate`, `vec_mat_mul`, `mat_mat_mul`, `vec_mat_mul_i8`, `mat_mat_mul_i8`, `rms_norm`, `softmax` — reusable for vision.
- `core.mojo`: `_init_model_impl_mojo`, `step_mojo`, `generate_embeddings_mojo`, `free_arena`, `reset_cache` — no vision FFI.
- `hydration.py`: `ImageHydrator` — loads images via PIL, returns raw RGB uint8 arrays. No preprocessing, no video.
- `backends.py`: `process_images()` on `CoreBackend` calls `_core.process_image()` per image — but `process_image` is not exported from core.mojo.
- `model.py`: `generate_stream()` calls `ImageHydrator().hydrate()` then `_backend.process_images()` before prefill.

**What was deleted in Ch3/Ch4:**
- `vision_ops.mojo` — deleted entirely (normalize_rgb, bilinear_resize, extract_patches, add_positional_embeddings)
- `VisionModelWeights`, `VisionLayerWeights` — deleted from model.mojo
- `process_image_mojo` — deleted from core.mojo

**Everything for vision must be built from scratch.**

## Architecture

```
Raw Image (any size)
    ↓
[Python] Select token budget → resize to target resolution (PIL bicubic)
[Python] Normalize pixels (SigLIP: (x/255 - 0.5) / 0.5)
[Python] Extract patches: reshape [H,W,3] → [N, 16*16*3]
    ↓ numpy float32 array via FFI
[Mojo] Patch embedding: patches @ embed_weight → [N, vision_hidden]
[Mojo] Add learned position embeddings
[Mojo] N vision transformer layers (bidirectional attention, GELU MLP)
[Mojo] Post-LayerNorm
[Mojo] Average pooling (3×3 kernel, stride 3) → reduce tokens by 9×
[Mojo] Vision projection → [num_tokens, decoder_hidden]
    ↓ stored in llm state
[Python] During prefill, replace <image> placeholder tokens with vision embeddings
```

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Image preprocessing | Python (PIL + numpy) | Already a `mogemma[vision]` dep. Not latency-critical. |
| Vision encoder forward | Mojo | 27 transformer layers is latency-sensitive. Reuses ops.mojo matmul/softmax. |
| GELU activation | New in ops.mojo | SigLIP uses standard GELU, not GEGLU. Same math as GEGLU gate half. |
| Average pooling | Mojo (new op) | Reduces patch count by 9×. Simple mean over 3×3 grid cells. |
| Variable resolution | Python token-budget table | 5 budgets map to target resolutions. Python picks closest. |
| Video frames | Python + ffmpeg subprocess | No new Python deps. ffmpeg ubiquitous. Each frame is independent image. |
| Token merging | New `step_with_embedding` FFI | Prefill replaces placeholder token steps with direct embedding injection. |

## Constraints

1. **Zero new dependencies.** PIL already optional via `mogemma[vision]`.
2. **One encoder struct** parameterized by layer count (16 or 27).
3. **Bidirectional attention.** No causal mask in vision encoder — every patch attends to every patch.
4. **Config-driven.** Vision params from `config.json`'s `vision_config` section.

---

## Implementation Plan

### Phase 1: Vision Weight Structs & Loading

#### Task 5.1: VisionLayerWeights and VisionModelWeights in model.mojo

Add vision encoder weight structs to `model.mojo`.

**Files:** `src/mo/mogemma/model.mojo`

**Structs:**
```
VisionLayerWeights:
  q_proj, k_proj, v_proj, o_proj: TensorInfo     # Self-attention
  fc1, fc2: TensorInfo                             # MLP (GELU, not GEGLU — single gate)
  layer_norm1, layer_norm2: TensorInfo             # Pre-attn, pre-MLP norms
```

```
VisionModelWeights:
  patch_embedding: TensorInfo     # [vision_hidden, patch_size*patch_size*3]
  position_embedding: TensorInfo  # [max_patches, vision_hidden]
  post_norm: TensorInfo           # Final LayerNorm
  projection: TensorInfo          # [decoder_hidden, vision_hidden]
  layers: List[VisionLayerWeights]
```

**HuggingFace tensor names:**
```
vision_tower.vision_model.embeddings.patch_embedding.weight
vision_tower.vision_model.embeddings.position_embedding.weight
vision_tower.vision_model.encoder.layers.{i}.self_attn.{q,k,v,out}_proj.weight
vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight
vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.weight
vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight
vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight
vision_tower.vision_model.post_layernorm.weight
multi_modal_projector.linear.weight
```

**Tests:** Struct construction, field count, embedding helper.

---

#### Task 5.2: Vision weight building in core.mojo

Extend `_init_model_impl_mojo` to load vision weights when `num_vision_layers > 0`.

**Files:** `src/mo/mogemma/core.mojo`

**Details:**
- New `_build_vision_weights(metadata, num_layers) → VisionModelWeights` — extracts `vision_tower.*` tensors
- Heap-allocate VisionModelWeights alongside KVCache/RoPETables
- Flatten/hydrate via Appender/Hydrator (same pattern as text model)
- New architecture_overrides: `num_vision_layers`, `vision_hidden_size`, `vision_num_heads`, `vision_intermediate_size`, `image_token_id`
- Vision scratch arena: `vision_hidden * max_patches * 4` (~20MB for 1120 patches)

**Tests:** Init with vision overrides, verify heap pointers, weight count.

---

### Phase 2: Image Preprocessing (Python)

#### Task 5.3: Variable-resolution preprocessing in hydration.py

Rewrite `ImageHydrator` to produce preprocessed patch arrays for Gemma 4.

**Files:** `src/py/mogemma/hydration.py`

**Details:**
- Token budget table: `{70: (280,280), 140: (280,560), 280: (560,560), 560: (560,1120), 1120: (1120,1120)}`
- `select_token_budget(h, w, max_tokens=560) → (target_h, target_w, num_tokens)` — picks closest aspect-ratio match
- `preprocess_image(rgb_array, max_tokens) → ImageInput` dataclass:
  1. Resize to target (PIL bicubic)
  2. Normalize: `(pixel/255 - 0.5) / 0.5`
  3. Patch: reshape `[H,W,3]` → `[N, 16*16*3]` (N = (H/16)*(W/16))
  4. Return `ImageInput(patches: ndarray, grid_h: int, grid_w: int, num_tokens: int)`
- `hydrate()` returns `list[ImageInput]` instead of `list[ndarray[uint8]]`

**Tests:** Budget selection for various aspect ratios, patch counts, normalization range, edge cases.

---

#### Task 5.4: Video frame extraction

Add video support to `ImageHydrator`.

**Files:** `src/py/mogemma/hydration.py`

**Details:**
- Detect video by extension (.mp4, .webm, .avi, .mov, .gif)
- Use `ffmpeg` subprocess: extract frames to temp dir at 1 FPS, max 32 frames, max 60s
- Each frame → `preprocess_image(frame, max_tokens=70)` (lowest budget)
- Fallback error if no ffmpeg: clear message
- Return `list[ImageInput]` — one per frame

**Tests:** Frame count limits, video detection, error without ffmpeg.

---

### Phase 3: Vision Encoder (Mojo)

#### Task 5.5: gelu and average_pool_2d in ops.mojo

Add two new ops needed for vision.

**Files:** `src/mo/mogemma/ops.mojo`

**Details:**
- `gelu[nelts](out_ptr, x_ptr, size)`: `0.5 * x * (1 + erf(x / sqrt(2)))` — same math as GEGLU gate, standalone.
- `average_pool_2d(out_ptr, x_ptr, grid_h, grid_w, hidden_size, kernel)`: mean over non-overlapping kernel×kernel blocks. Output: `[grid_h/kernel, grid_w/kernel, hidden_size]`.

**Tests:** GELU against known values, pooling reduction factor, pooling with non-divisible grids (pad).

---

#### Task 5.6: Bidirectional vision attention in layers.mojo

New batched bidirectional attention for vision transformer.

**Files:** `src/mo/mogemma/layers.mojo`

**Details:**
- `forward_vision_attention(out_ptr, x_ptr, weights: VisionLayerWeights, num_tokens, hidden_size, num_heads, head_dim, scratch_ptr)`:
  - Q, K, V projections via `mat_mat_mul` (batched — all patches at once)
  - Attention: `softmax(Q @ K^T / sqrt(head_dim))` — **NO causal mask**
  - Output projection
- Key difference from decoder: no KV cache, no position masking, full bidirectional, all tokens processed simultaneously.

**Tests:** Known-vector test, verify attention matrix is NOT lower-triangular.

---

#### Task 5.7: Vision encoder forward pass

Full SigLIP encoder: patches → vision tokens.

**Files:** `src/mo/mogemma/layers.mojo`

**Details:**
- `forward_vision_layer(out_ptr, x_ptr, weights: VisionLayerWeights, ...)`:
  1. LayerNorm1 → bidirectional attention → residual
  2. LayerNorm2 → MLP (fc1 → GELU → fc2, NOT GEGLU) → residual
- `forward_vision_encoder(out_ptr, patches_ptr, weights: VisionModelWeights, num_patches, grid_h, grid_w, vision_config...)`:
  1. Patch embedding: `patches @ patch_embed_weight^T`
  2. Add position embeddings
  3. Loop N vision layers
  4. Post-LayerNorm
  5. Average pooling (3×3)
  6. Vision projection → decoder hidden dim

**Tests:** 1-layer encoder with synthetic weights, pooling token reduction, projection dim.

---

### Phase 4: FFI & Token Merging

#### Task 5.8: process_image and step_with_embedding FFI

Two new FFI entrypoints in core.mojo.

**Files:** `src/mo/mogemma/core.mojo`

**Details:**
- `process_image_mojo(llm, patches_obj, grid_h, grid_w)`:
  1. Read patches (numpy float32 `[N, patch_dim]`)
  2. Hydrate VisionModelWeights from heap
  3. Call `forward_vision_encoder`
  4. Store vision embeddings in `llm["vision_embeddings"]` as list
  5. Return num_tokens
- `step_with_embedding_mojo(llm, embedding_obj, temp, top_k, top_p)`:
  1. Instead of token lookup, use provided embedding vector directly
  2. Feed through decoder layers (same as `step_mojo` minus embed_tokens lookup)
  3. Return logits
- Register both in `PyInit__core`

**Tests:** Round-trip with mock weights, output shapes.

---

#### Task 5.9: Token merging in generate_stream

Replace placeholder tokens with vision embeddings during prefill.

**Files:** `src/py/mogemma/model.py`

**Details:**
- In `generate_stream()`, after tokenizing:
  1. Find `<image>` placeholder token positions (from `image_token_index` in config)
  2. For each image: `_backend.process_image(llm, image.patches, image.grid_h, image.grid_w)`
  3. During prefill loop: when hitting placeholder, call `step_with_embedding` with vision token instead of normal `step`
  4. Position IDs: vision tokens get sequential positions
- `_parse_gemma4_architecture` reads `image_token_index` from config.json
- Update `CoreBackend.process_images()` to call `process_image` per image with correct args

**Tests:** Placeholder detection, vision token count matching, position continuity.

---

#### Task 5.10: Config wiring

Wire vision config from config.json through to Mojo.

**Files:** `src/py/mogemma/model.py`, `src/py/mogemma/config.py`

**Details:**
- `_parse_gemma4_architecture` reads from config.json:
  - `vision_config.num_hidden_layers` → `num_vision_layers`
  - `vision_config.hidden_size` → `vision_hidden_size`
  - `vision_config.num_attention_heads` → `vision_num_heads`
  - `vision_config.intermediate_size` → `vision_intermediate_size`
  - `vision_config.patch_size` (verify = 16)
  - `image_token_index` → `image_token_id`
- `GenerationConfig` gains `max_image_tokens: int = 560`

**Tests:** Config parsing with/without vision section, defaults.

---

### Phase 5: Cleanup & Verification

#### Task 5.11: Delete legacy vision assumptions and run full suite

**Files:** Multiple

**Details:**
- Verify no 384×384 or patch_size=14 references remain
- Update `backends.py` — `process_image` is now real, not optional hasattr check
- Full test suite green

**Verification:**
```bash
CI=true uv run pytest -x -v
grep -r "384\|patch_size.*14" src/
```

---

## Files Modified

| File | Changes |
|---|---|
| `src/mo/mogemma/model.mojo` | Add VisionLayerWeights, VisionModelWeights |
| `src/mo/mogemma/layers.mojo` | Add forward_vision_attention, forward_vision_layer, forward_vision_encoder |
| `src/mo/mogemma/ops.mojo` | Add gelu, average_pool_2d |
| `src/mo/mogemma/core.mojo` | Add process_image_mojo, step_with_embedding_mojo, vision weight building |
| `src/py/mogemma/hydration.py` | Rewrite: variable resolution, normalization, patches, video |
| `src/py/mogemma/model.py` | Vision config parsing, token merging in generate_stream |
| `src/py/mogemma/config.py` | Add max_image_tokens |
| `src/py/mogemma/backends.py` | process_image becomes concrete |

## New Test Files

| File | Purpose |
|---|---|
| `src/py/tests/test_vision_preprocessing.py` | Resize, normalize, patch extraction, token budget |
| `src/py/tests/test_video_frames.py` | Video frame extraction, limits |
| `src/py/tests/test_vision_config.py` | Vision config.json parsing |
| `src/mo/tests/test_vision_encoder.mojo` | Vision attention, encoder forward, pooling |

## Task Summary

| Task | Description | Status |
|---|---|---|
| 5.1 | VisionLayerWeights + VisionModelWeights structs | [x] 8658969 |
| 5.2 | Vision weight building in core.mojo | [x] c00625c |
| 5.3 | Variable-resolution image preprocessing | [x] daa050f |
| 5.4 | Video frame extraction | [x] cb3abbe |
| 5.5 | gelu + average_pool_2d ops | [x] b14cbbc |
| 5.6 | Bidirectional vision attention | [x] 545fb59 |
| 5.7 | Vision encoder forward pass | [x] cec2e6e |
| 5.8 | process_image + step_with_embedding FFI | [x] 5569744 |
| 5.9 | Token merging in generate_stream | [x] 959c163 |
| 5.10 | Config wiring | [x] 0ec1e31 |
| 5.11 | Cleanup + full suite | [x] f8a6d33 |
