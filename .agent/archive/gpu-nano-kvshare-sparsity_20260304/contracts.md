# KV-Share and Sparsity Contracts (Phase 1 Checkpoint)

## 1.1 KV-Share Start Detection
- **Definition**: The runtime scans Nano layer parameters at initialization. `kv_share_start` is the integer index of the first layer where `k_proj` and `v_proj` weights are effectively zero (sum of squares is `<= 1e-8`).
- **Contract**: This value is calculated exactly once during `_init_model_impl_mojo` and stored as an immutable integer in the `llm` dictionary. It must yield identical results across all backends.

## 1.2 Layer Remapping Truth Table
During the Nano step or sequence forward pass, the actual KV cache index written to/read from differs from the architectural layer index:
- **Pre-share** (`l < kv_share_start`): `write_kv = True`, `kv_layer_idx = l`.
- **Shared-full** (`l >= kv_share_start` and `(l + 1) % 5 == 0`): `write_kv = False`, `kv_layer_idx = last_full_kv_layer`.
- **Shared-sliding** (`l >= kv_share_start` and `(l + 1) % 5 != 0`): `write_kv = False`, `kv_layer_idx = last_sliding_kv_layer`.
*(Note: If no valid prior layers exist, `kv_layer_idx` clamps to `kv_share_start - 1`)*.

## 1.3 Activation Sparsity Contract
- **Definition**: Nano MLP outputs (the `gate` layer) are subject to activation sparsity using a dynamic z-score threshold.
- **Rules**:
  1. Sparsity is only computed and applied for the first 10 decoder layers.
  2. The cutoff is computed as `mean + std * 0.95`.
  3. Any activation strictly `< cutoff` is forced to `0.0`.
  4. Reductions (mean/std) must be deterministic. Tolerances for parity should be `atol <= 1e-4` to handle minor floating point associative reduction differences on the GPU, though strict deterministic paths are preferred.

## 1.4 Signoff
These behaviors are hard requirements for `gemma3n-e2b-it` correctness and must be fully mirrored onto the GPU dispatch boundaries before performance benchmarks are authorized.