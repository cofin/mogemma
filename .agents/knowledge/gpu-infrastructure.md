# GPU Backend & Infrastructure

The GPU backend is built on Mojo's native `std.gpu.host` API, supporting high-performance inference on accelerators (CUDA/ROCm).

## Architecture

- **Backend Traits:** Compute operations are abstracted through a `ComputeBackend` trait. CPU implementations are standard, while GPU-enabled builds use specialized `GPUBackend` implementations gated by `comptime if has_accelerator()`.
- **Resource Management:** `GPUContext` manages device state, including persistent weight buffers, a unified KV cache arena, and scratch space for intermediate tensors.
- **FFI Boundary:** GPU-allocated resources (e.g., buffer pointers) are passed between Python and Mojo as integer addresses. Mojo uses `UnsafePointer[T, MutExternalOrigin]` for access.

## Kernel Implementation

Kernels are developed using Mojo's GPU DSL.

- **Element-wise:** ReLU, GeGLU, Softmax, and RoPE rotations are implemented as device functions launched across a grid.
- **Reductions:** `rms_norm` and `softmax` use warp- and block-level primitives (e.g., `block_max`, `block_sum`) for efficient computation within a workgroup.
- **Matmul:** Vector-matrix and matrix-matrix multiplications use tiling and shared memory (`AddressSpace.SHARED`) to maximize throughput.

## Forward Path Architecture

All forward functions are parameterized on `[B: ComputeBackend]` with additional `S: AnyType, C: AnyType, P: AnyType` type params for weight streaming. Key separation:

- **Step/Encoder Functions** (e.g., `forward_gemma4_step`, `forward_vision_encoder`): Own weight streaming orchestration. Use `comptime if has_accelerator()` → `rebind` to cast erased types back to `WeightStage`/`GPUContext`. Upload per-layer weights, sync, then call layer functions.
- **Layer Functions** (e.g., `forward_gemma4_layer`, `forward_vision_layer`): Completely backend-agnostic. Receive device pointers via `LayerWeights`/`VisionLayerWeights` structs, dispatch all ops through `B.rms_norm()`, `B.vec_mat_mul()`, etc.
- **CPU Callers**: Pass `S=Int, C=Int, P=Int` with dummy `0` values — zero overhead.

### Weight Tiers
- **Persistent** (`PersistentBuffers` in Mojo): `embed_tokens`, `lm_head`, and final `norm` — these are uploaded once at initialization and kept on the device for the entire session. The `PersistentBuffers` struct holds device pointers and element counts for these weights.
- **Streamed** (`WeightStage` in Mojo): Per-layer weights (projections, MLP, layer norms). These are packed into a pinned host buffer, copied to a reusable device staging buffer each step, and consumed by the layer kernels. This "streaming" minimizes device memory consumption for weights.
- **Encoder Layers**: Vision/audio layers are also streamed per-layer via specialized upload functions.

### GPU Dispatch (core.mojo)
- `step_mojo` checks `_gpu_initialized` flag, routes to `_run_step[GPUBackend, GPUKVCache, ...]` or `_run_step[CPUBackend, KVCache, ...]`.
- Same pattern for `step_with_embedding_mojo`, `process_image_mojo`, `generate_embeddings_mojo`.
- GPU handles stored as heap-allocated pointers in `llm` dict, retrieved inline via `UnsafePointer[T](unsafe_from_address=Int(py=llm["_key"]))`.
- `_init_gpu_resources()` creates all GPU objects; `_cleanup_gpu_resources()` frees in reverse order.

### Encoder Weight Sharing
- Vision and audio encoders both use `upload_vision_layer_weights` and `VisionLayerWeights` struct — audio layers have identical weight shapes.
- Audio encoder GPU dispatch is a stub (both CPU and GPU paths) — blocked on HuggingFace tensor name standardization for audio weight hydration.

### Device-to-Device Copy
- `copy_state` in `gpu_context.mojo` uses a kernel launch (not `enqueue_copy`) because the scratch buffer uses pointer arithmetic within a single `DeviceBuffer`, not separate handles.
- For `generate_embeddings` GPU path, hidden states are read directly from GPU scratch pointer — works for v1 because hidden_size is small (~16KB for 31B).

## Optimization Patterns

- **Contiguous Allocation:** All GPU-side KV cache layers are stored in a single large buffer to reduce host-to-device communication and allocation overhead.
- **Weight Staging:** Per-layer weights are packed into a pinned host buffer, copied to a reusable device staging buffer, then consumed by layer kernels. One host→device copy per layer.
- **Compile-Time Gating:** Extensive use of `comptime if` ensures that GPU-specific code paths are only included in builds targeting accelerators, maintaining zero overhead for CPU-only builds.
- **State Swap:** Layer loop state swap uses `backend.copy()` (GPU kernel) instead of scalar loops for device compatibility.
