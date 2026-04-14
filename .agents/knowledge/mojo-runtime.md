# Mojo runtime (`src/mo/mogemma/`)

## File roles

| File | Owns |
|---|---|
| `core.mojo` | FFI entry points (`step_mojo`, `init_model`, `load_weights`, ...), backend dispatch, session state |
| `layers.mojo` | Forward-pass kernels — backend-agnostic via `ComputeBackend` trait |
| `model.mojo` | Struct definitions: `TensorInfo`, `*LayerWeights`, `*ModelWeights` |
| `gpu_context.mojo` | `GPUContext`, `WeightStage`, upload/packer functions |
| `ops.mojo` | `ComputeBackend` trait + `CPUBackend` implementation |
| `ops_gpu.mojo` | `GPUBackend` implementation, guarded by `has_accelerator()` |

## Core structs

### `TensorInfo`

- Holds weight pointer + shape + quantization metadata.
- Fields: `ptr: UnsafePointer[Float32, MutExternalOrigin]`, `i8_ptr`, `scale_ptr` (for int8 quant), `is_quantized: Bool`, shape integers.

### `LayerWeights` (base transformer layer)

- Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `q_norm`, `k_norm`.
- MLP: `gate_proj`, `up_proj`, `down_proj`.
- Norms: `pre_attention_norm`, `post_attention_norm`, `pre_ffw_norm`, `post_ffw_norm`.

### `PLELayerWeights`

- `per_layer_embedding`, `per_layer_projection`, `per_layer_norm`.

### `MoELayerWeights` (contract being revised in `fix/timeout` branch)

Target design for Gemma 4 MoE:
- Attention + two sets of MLP weights (dense branch + routed experts).
- Dense branch: `dense_gate_proj`, `dense_up_proj`, `dense_down_proj`.
- Router: `router_proj`, `router_scale`, `per_expert_scale`.
- Experts packed: `expert_gate_up_proj` `[E, 2·I_moe, H]`, `expert_down_proj` `[E, H, I_moe]`.
- Six norms: `pre_attention`, `post_attention`, `pre_ffw` (dense pre), `post_ffw1` (dense post), `pre_ffw2` (MoE pre), `post_ffw2` (MoE post), plus optional `post_ffw` of unclear role.
- `skip_scale` `(1,)` scalar.

### `VisionLayerWeights`

- Attention + GEGLU MLP.
- Includes `fc1_up` alongside `fc1` and `fc2` — vision uses GEGLU, not plain MLP.

### `ModelWeights` / `MoEModelWeights` / etc.

- Arrays of per-layer struct instances.
- Global tensors: `embed_tokens` (aka `lm_head`), `final_norm`.

## Forward-pass dispatch

```
step_mojo(llm, input_ids, ...)               # FFI entry
  → _gpu_initialized?
      yes → _run_step[GPUBackend, GPUKVCache, S, C, P]
      no  → _run_step[CPUBackend, CPUKVCache, 0, 0, 0]
  → forward_step
  → forward_layer × num_hidden_layers
      → forward_sliding_attention | forward_full_attention
      → forward_mlp | forward_moe_layer | forward_vision_layer
```

- `S, C, P` are compile-time GPU handle keys. CPU callers pass `0` — zero overhead.
- `rebind` casts backend-erased types back to concrete `WeightStage` / `GPUContext` inside GPU branches.

## Weight tiers

| Tier | What | When uploaded |
|---|---|---|
| **Persistent** | `embed_tokens`, `lm_head`, `final_norm`, RoPE tables | Once, at init |
| **Streamed per-layer** | Per-layer attention + MLP weights | Per step, into reusable device staging buffer |
| **Encoder streamed** | Vision / audio per-layer | Per encoder invocation |

- Streamed path: pack into pinned host buffer → copy once to reusable device staging buffer → kernels consume directly.
- One host→device copy per layer.

## GPU context

- `GPUContext` holds device handles, staging buffers, scratch allocations.
- `WeightStage` is the per-layer packer: flatten layer weights into contiguous bytes, copy to device, reconstruct pointers on device.
- Scratch buffer uses pointer arithmetic within a **single `DeviceBuffer`** — no multi-buffer allocations per layer.

## Device-side operations

- **Device-to-device copy** uses kernel launch, NOT `enqueue_copy` (scratch arithmetic needs a single buffer).
- State swaps (e.g., prev/next hidden state buffer ping-pong) use `backend.copy()` — a kernel launch — not a scalar loop, for device compatibility.
- Embedding output: hidden states are read directly from the GPU scratch pointer by the Python side via the opaque handle.

## FFI conventions

- Python hands pointers as `int`. Mojo reconstructs via `UnsafePointer(unsafe_from_address=Int(py=...))`.
- Pointer origin type: `MutExternalOrigin` for Python-owned memory, `MutAnyOrigin` for inside-Mojo derived pointers.
- The `llm` dict pattern: opaque handles stored as `Int(heap_ptr)` in a Python dict; Mojo retrieves via `unsafe_from_address`.

## Backend trait

`ComputeBackend` exposes: `vec_mat_mul`, `mat_mat_mul`, `vec_mat_mul_i8`, `rms_norm`, `softmax`, `rope_rotate`, `gelu`, `geglu`, `copy`, plus reductions. `CPUBackend` is always available. `GPUBackend` is compiled in when `has_accelerator()` is `True`.

Layer kernels use:

```mojo
@always_inline
fn forward_sliding_attention[B: ComputeBackend, K: KVCacheTrait](
    mut backend: B, ...
):
```

→ one compiled variant per backend, zero dispatch overhead.

## Gotchas

- **GC corruption:** If the Python `numpy.ndarray` whose pointer Mojo holds goes out of scope, the weights vanish mid-forward. Keep references on the Python side for the model's lifetime.
- **Alignment:** When casting raw pointers, verify 64-bit alignment. Some TensorStore-derived arrays need `np.ascontiguousarray()` before pointer extraction.
- **Safetensors mmap lifetime:** Holding a `TensorInfo` that points into an mmap'd safetensors shard means the `safe_open` file handle must stay open.
