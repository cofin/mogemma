# Phase 1: Correctness and Performance Gate Contract

## 1.1 Correctness Dimensions
- **Primitive Outputs**: CPU vs GPU kernels (`vec_mat_mul`, `rms_norm`, `rope_rotate`, `softmax`, `geglu`) must match identically for shapes and maintain an absolute tolerance (`atol`) <= 1e-6.
- **Layer Outputs**: The outputs of `forward_attention` and `forward_mlp` must maintain `atol <= 1e-4` given identical pre-populated KV cache states and prompt embeddings.
- **Logits / `_core.step`**: `out_logits` must be evaluated pre-sampling to ensure identical shape, identically matching `argmax` token, and numeric `atol <= 1e-4` against CPU.
- **End-to-End Deterministic Generation**: When using `temperature=0.0` and `top_k=1`, the generated token sequence for the identical prompt must match 100% token-for-token.

## 1.2 Deterministic Execution Policy
- **Prompt Corpus**: Fixed set of instruction-formatted inputs representing short (<= 64 tokens), medium (~512 tokens), and long (>= 1024 tokens) context prefill.
- **Decoding**: Pure greedy decoding (`temperature=0.0`, `top_k=1`, `top_p=1.0`).
- **Reporting**: The session must explicitly assert `step_backend` is consistently either `cpu` or `cuda` for the duration of generation, logging any `fallback_reason`. Mixed backend generation in a single request invalidates the test run.

## 1.3 Baseline Collection Requirements
Performance reporting must include the following metadata fields to be considered valid release evidence:
- **Git SHA**: The current commit hash.
- **Mojo/Python Versions**: E.g., `Mojo 0.26.1`, `Python 3.10.18`.
- **Hardware**: CPU identifier and GPU identifier (e.g. `NVIDIA L4`).
- **Drivers**: CUDA runtime version or MAX Engine version.
- **Corpus Metadata**: `prompt_length`, `max_new_tokens`.

## 1.4 Checkpoint Signoff
- **Correctness Thresholds**: `atol <= 1e-6` (primitives), `atol <= 1e-4` (layers/logits), `100%` (tokens).
- **Throughput Thresholds**: Median generation tokens/sec must be >= 1.25x the CPU baseline on the designated validation host.
- **Latency Thresholds**: Short-prompt p95 latency must not regress > 15%.