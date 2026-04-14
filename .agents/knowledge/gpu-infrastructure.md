# GPU infrastructure

## Backend abstraction

- `ComputeBackend` trait in `ops.mojo`: `vec_mat_mul`, `mat_mat_mul`, `vec_mat_mul_i8`, `rms_norm`, `softmax`, `rope_rotate`, `gelu`, `geglu`, `copy`, reductions.
- `CPUBackend` is always compiled in. `GPUBackend` is conditional on `has_accelerator()`.
- Layer kernels are parametric on `B: ComputeBackend` — one compiled variant per backend, zero dispatch overhead.
- CPU-only builds pay **no GPU overhead** (compile-time elision).

## Kernel set

- **Element-wise:** ReLU, GeGLU, Softmax (warp/block reductions), RoPE rotations.
- **Reductions:** `rms_norm` (warp + block), softmax (two-pass max/exp).
- **Matmul:** tiled w/ shared-memory staging. `vec_mat_mul` for batch=1 (generation step), `mat_mat_mul` for batch>1 (prefill, embeddings).
- **Int8 path:** `vec_mat_mul_i8` for quantized weights; scale_ptr carries per-row scales.

## KV cache

- **Single contiguous arena** across all layers. Preferred over per-layer allocation — minimizes host↔device comms and allocation churn.
- Ring buffer for sliding-window attention.
- Shared-KV layers (per `kv_sharing_layer_map`) skip writes to avoid double-booking.
- Allocated via `GPUKVCache`; CPU fallback `CPUKVCache`. Both implement `KVCacheTrait`.

## Weight staging

Three tiers, routed per lifetime:

| Tier | Uploaded | Buffer |
|---|---|---|
| **Persistent** | Once at init | Pinned on device for the session |
| **Streamed per-layer** | Per step | Reusable device staging buffer, overwritten layer-by-layer |
| **Encoder streamed** | Per encoder invocation | Same staging buffer, different packer |

Upload flow:

```
pinned_host_buffer ← pack(LayerWeights)          # on CPU
device_staging_buffer ← host-to-device copy       # one copy per layer
kernel(device_staging_buffer)                     # zero-copy consumption
```

- Packers live in `gpu_context.mojo`: `upload_layer_weights`, `upload_expert_weights`, `upload_moe_attention_weights`, `upload_vision_layer_weights`.

## Session memory model

- `GPUContext` owns: persistent buffers, per-step staging, KV arena, scratch, embedding scratch pool (optional).
- `_init_gpu_resources()` allocates all of these.
- `_cleanup_gpu_resources()` frees in reverse order.
- Buffer handles exposed to Python as integer pointers in the `llm` dict (opaque — never dereferenced on Python side).

## Device-memory visibility

Exposed to Python via `llm` dict:
- `pos` — current sequence position.
- `session_kv_cache_len`, `step_scratch_len`, `embedding_scratch_len` — sizes.
- GPU buffer pointers (as `Int`).
- Backend debug counters.

## Device selection

- Environment-variable-driven or API call into `_core.select_device(device_id)`.
- Device selection must happen **before** `_init_gpu_resources()` — changing devices mid-session is unsupported.

## CUDA build + distribution

- NVIDIA path requires CUDA runtime on target machine.
- Wheels built with CUDA enabled are labeled accordingly; CPU-only wheels exclude GPU kernels entirely (smaller artifact, no CUDA dep).
- See [build-and-packaging.md](build-and-packaging.md) for cibuildwheel quirks.

## Perf gates

- Benchmarks advisory only — do NOT gate PRs on perf (results vary with hardware).
- Baseline variance → threshold bands, not hard numbers.
- Hardware-specific tuning (tile sizes, block dims) must be documented next to the kernel, not in knowledge base (it's code-adjacent).

## Gotchas

- **Scratch buffer arithmetic:** pointer math must stay within one `DeviceBuffer`. Multiple `DeviceBuffer` allocations per layer would require `enqueue_copy` for inter-buffer moves — explicitly avoided.
- **State swaps:** use `backend.copy()` (kernel launch), not scalar loops, for device compatibility. CPU path elides this to a `memcpy`.
- **Embedding scratch:** when embedding output is returned to Python, the GPU scratch pointer is handed over directly. Do not reuse the scratch for the next call until Python consumes it.
- **`has_accelerator()`:** `@comptime if` gating avoids any GPU symbols from entering the CPU-only build.
