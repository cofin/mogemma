# GPU Standard Kernels: Kernelization Design (Phase 2 Checkpoint)

## 2.1 Grouped/Fused GPU Kernel Coverage

| Primitive Operation | GPU Kernel Strategy | Notes |
| --- | --- | --- |
| Q/K/V Projections | Batched/Unrolled `vec_mat_mul` | In Phase 1, we will map standard `vec_mat_mul` to a single matrix-vector multiply GPU kernel launch. Fusing all three is deferred to avoid altering `layers.mojo` structure too deeply yet. |
| Q/K Norm | Element-wise `rms_norm` GPU kernel | Uses the standard `(1 + weight)` behavior. Evaluated on head basis. |
| RoPE (Rotary Positional Embedding) | Element-wise `rope_rotate` GPU kernel | Launched per-head on Q and K buffers. |
| Score Softmax | Fused Vector Softmax GPU kernel | Over the sequence length (up to `pos+1`), reducing over the trailing dimension. |
| Value Accumulation | Dot-product/Vector MAC GPU kernel | Computes the weighted sum of `v_head_ptr` over sequence tokens. |
| Output Projection | `vec_mat_mul` GPU kernel | Matrix-vector multiply back to `hidden_size`. |
| RMSNorm (Pre/Post) | Row-wise `rms_norm` GPU kernel | Keeps `(1+weight)` rule, operating over `hidden_size`. |
| GEGLU | Element-wise `geglu` GPU kernel | Element-wise operations on Gate and Up projections, mapping `intermediate_size`. |

## 2.2 Scratch-Memory Contract

To prepare for Chapter 2 (Device-Resident memory):
- **Attention/Layer Contract**: `forward_attention` and `forward_mlp` accept exactly ONE `scratch_ptr: UnsafePointer[Float32]`.
- This pointer represents a sufficiently large contiguous slab pre-allocated for the token step by `core.mojo`.
- **Layout expectations**: 
  - `q_ptr`, `k_ptr`, `v_ptr` packed at the front of `attn_scratch`.
  - `attn_out_ptr` offset after QKV space.
  - `scores_ptr` offset after `attn_out_ptr`.
- GPU kernels will receive offsets within this single scratch buffer. In Chapter 2, this `scratch_ptr` will simply transition from being host-allocated to device-allocated, requiring ZERO changes to the kernel launch signatures.

## 2.3 Standard MLP Execution Plan

- **Inputs**: `x_ptr` (normalized residual), `scratch_ptr`.
- **Step 1 (Gate & Up Projections)**: Launch two `vec_mat_mul` kernels or one batched launch. Outputs stored in `scratch_ptr[0:intermediate]` and `scratch_ptr[intermediate:2*intermediate]`.
- **Step 2 (GEGLU)**: Launch `geglu` element-wise kernel on the two vectors, storing output into `scratch_ptr[2*intermediate:3*intermediate]`.
- **Step 3 (Down Projection)**: Launch `vec_mat_mul` from GEGLU output back to `hidden_size`, storing in `out_ptr`.
- **CPU Fallback**: The exact same plan is followed in CPU fallback mode using `ops.mojo`. If any GPU primitive fails to initialize, the entire session must fall back to the CPU execution loop.

## 2.4 GPU Implementation vs CPU-Only Rationale

Every standard operation reached from `forward_step` has been mapped.
- `vec_mat_mul`: GPU dispatch path planned.
- `rms_norm`: GPU dispatch path planned.
- `rope_rotate`: GPU dispatch path planned.
- `softmax`: GPU dispatch path planned.
- `geglu`: GPU dispatch path planned.
- Cache KV writing: **CPU pointer loop** logic preserved. For Phase 1, we will copy computed K/V data into the host-side KV cache via pointer math. Moving this to a GPU indexing kernel is explicitly reserved for Chapter 2 when KV arrays move to the device.
