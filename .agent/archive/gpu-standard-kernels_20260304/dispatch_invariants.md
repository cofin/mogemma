# GPU Standard Kernels: Dispatch & Invariants Table
(Phase 1 Checkpoint)

## 1.1 Call Graph Mapping

**Standard Dispatch Trace:**
- `core.mojo:step_mojo`
  - `core.mojo:_forward_step_standard_runtime`
    - `layers.mojo:forward_layer` (in a loop)
      - `ops.mojo:rms_norm` (input_layernorm)
      - `layers.mojo:forward_attention`
        - `ops.mojo:vec_mat_mul` (q_proj, k_proj, v_proj, o_proj)
        - `ops.mojo:rms_norm` (q_norm, k_norm)
        - `ops.mojo:rope_rotate` (RoPE on q and k)
        - KV cache writes (pointer offset logic)
        - `ops.mojo:softmax` (scores)
        - Attention value accumulation (pointer loops)
      - `ops.mojo:rms_norm` (post_attention_layernorm)
      - `ops.mojo:rms_norm` (pre_feedforward_layernorm)
      - `layers.mojo:forward_mlp`
        - `ops.mojo:vec_mat_mul` (gate_proj, up_proj)
        - `ops.mojo:geglu`
        - `ops.mojo:vec_mat_mul` (down_proj)
      - `ops.mojo:rms_norm` (post_feedforward_layernorm)
    - `ops.mojo:rms_norm` (final model.norm)
    - `ops.mojo:vec_mat_mul` (lm_head projection for logits)

**Explicit Non-Goals:**
- Nano paths: `forward_nano_step`, `forward_attention_nano`, `forward_mlp_nano`, `_apply_nano_activation_sparsity`, AltUp/Laurel projections.
- Embedding/Prefill paths: `forward_sequence`, `forward_nano_sequence`.
- No modifications to the public `src/py/mogemma/model.py` API structure.

## 1.2 Ownership Boundaries
- **`core.mojo`**: Owns the session backend latch, backend capability querying, and the CPU fallback policy. Resolves whether standard step will use CPU paths or GPU kernel paths based on Python's `device_selection` object.
- **`layers.mojo`**: Owns the standard graph orchestration. Defines the token-step topology. Will dispatch either to CPU primitives or to GPU primitives based on the latch provided by `core.mojo`.
- **`ops.mojo` (or GPU sibling like `ops_gpu.mojo`)**: Holds the backend-specific kernel implementations (launch configurations, threads, shared memory). Must expose a stable signature that does not leak backend details back into `layers.mojo`.

## 1.3 Kernel-Entry Invariants
- **Contiguous Pointers:** All primitives accept strictly contiguous `UnsafePointer[Float32]` memory blocks. No strided tensor views at the kernel entry level.
- **Scratch Layout:** `forward_attention` and `forward_mlp` must be given one contiguous scratch space pre-allocated by `core.mojo`, compatible with later device-resident pools.
- **Cache Pointers:** KV pointers (`kv_cache_k_ptr`, `kv_cache_v_ptr`) remain stable per step. The indexing is strictly layer-major.
- **Logits Shape:** 1D float32 output buffer, `vocab_size` length, unnormalized.
- **Prohibited Data:** No passing of AltUp/Laurel specific weights or `per_layer_dim` sizes into the standard dispatch kernel boundary.

## 1.4 Dispatch & Invariants Table (Checkpoint)

| Standard Symbol / Logic | Implementation Strategy for this Chapter | Fallback/CPU State |
|-------------------------|------------------------------------------|--------------------|
| `step_mojo`             | Backend check/latch, route to standard | Preserved (Default)|
| `_forward_step_standard_runtime` | Thread backend handle/scratch to layers | Preserved |
| `forward_layer`         | Dispatch to GPU vs CPU versions of att/mlp| Preserved |
| `forward_attention`     | GPU dispatch for fused or unrolled QKV/RoPE/Softmax | Preserved |
| `forward_mlp`           | GPU dispatch for Gate/Up/GEGLU/Down | Preserved |
| `vec_mat_mul`           | GPU matrix-vector or slim matrix-matrix kernel | `ops.mojo` |
| `rms_norm`              | GPU row-wise reduction kernel (keeps `1+weight` rule)| `ops.mojo` |
| `rope_rotate`           | GPU element-wise inplace rotate kernel | `ops.mojo` |
| `softmax`               | GPU vector softmax (over past context limit) | `ops.mojo` |
| `geglu`                 | GPU element-wise activation kernel | `ops.mojo` |
| Cache KV writes         | Ported to GPU indexing kernel or fused into Attention | CPU pointer loops|
| Nano / AltUp            | Out of scope, untouched                  | Preserved |
| `forward_sequence`      | Out of scope, untouched                  | Preserved |
