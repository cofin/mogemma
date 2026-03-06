# Nano GPU Path Contracts (Phase 1 Checkpoint)

## 1.1 Explicit Kernel Contracts

| Operation | Inputs | Outputs | Precomputed/Weights | Parity Target | Notes |
|---|---|---|---|---|---|
| **Nano Attention** | `x_ptr` (hidden_size), `scratch_ptr` | `out_ptr` (hidden_size) | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `q_norm` (weighted), `k_norm` (weighted), `v_norm` (unit), RoPE | `atol <= 1e-4` | Must preserve `scale=1.0` (unlike standard). Must use `(w * x)` for weighted QK norms. Must support optional KV writes based on `write_kv` flag. |
| **AltUp Predict** | `x_ptr` (hidden_size) | `gate_out`, `up_out` | `router_weight`, `gate_proj`, `up_proj` | `atol <= 1e-4` | Operates across the specific `num_modalities` loop. Modality sorting is strictly enforced. |
| **AltUp Correct** | `x_ptr`, `gate_in`, `up_in` | `out_ptr` | `router_weight`, `down_proj` | `atol <= 1e-4` | Combines outputs of predictor. Uses identical layout/tensor constraints. |
| **Laurel** | `x_ptr` | `out_ptr` | `laurel_weight` | `atol <= 1e-4` | Stream-mixing step that precedes or follows attention. Must remain ordered. |
| **Per-Layer Map** | `x_ptr`, `layer_idx` | `out_ptr` | `gate`, `up`, `down` tensors | `atol <= 1e-4` | Uses `per_layer_dim` defined at runtime. Does not assume a hardcoded static size. |

## 1.2 GPU Dispatch Seam
- **Session Latch**: `core.mojo:step_mojo` already features `step_backend` evaluation. Nano execution will route through `_forward_step_nano_runtime`, which will be split to call `_forward_step_nano_gpu_runtime` if `step_backend == "cuda"`.
- **Public Python API**: Remains completely unchanged. The `device_selection` object seamlessly directs initialization to the GPU fallback logic or device-resident handles.

## 1.3 Ownership Table
| CPU Function | GPU Equivalent | Module | Fallback Policy |
|---|---|---|---|
| `forward_attention_nano` | `forward_attention_nano_gpu` | `layers.mojo` | Fail fast if CUDA not present. |
| `forward_altup_predict` | `forward_altup_predict_gpu` | `layers.mojo` | Same as above. |
| `forward_altup_correct` | `forward_altup_correct_gpu` | `layers.mojo` | Same as above. |
| `forward_laurel` | `forward_laurel_gpu` | `layers.mojo` | Same as above. |
| `forward_per_layer_mapping` | `forward_per_layer_mapping_gpu` | `layers.mojo` | Same as above. |
