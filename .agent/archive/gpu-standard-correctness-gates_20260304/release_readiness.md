# Phase 4: Release Readiness and Manual Validation

## 4.1 CI Integration Points and Triage Workflow
- **CI Triggers**: The parity test suite (`test_ops.mojo`, `test_layers.mojo`, `test_mojo_core.py`, `test_gemma_model.py`) runs on every pull request that modifies standard generation paths.
- **Performance Smoke Tests**: `tools/benchmark.py --corpus short --rounds 3` runs in CI to verify the threshold gating logic doesn't crash, but pure performance metrics are not enforced in standard CI (due to variable runner hardware).
- **Triage Workflow**:
  - *Correctness Failure*: Treat as P0 blocker. Do not merge.
  - *Fallback Failure*: Ensure Python properly surfaces the underlying hardware rejection (e.g. `cuda_kernels_unimplemented` or `vram_exhausted`).
  - *Performance Failure*: Investigated on the release host prior to tag.

## 4.2 Rollback / Default-Disable Criteria
- If `atol > 1e-4` on layer outputs or logits, standard GPU is default-disabled.
- If tokens differ (`temperature=0.0`), standard GPU is default-disabled.
- If median throughput fails to meet the `1.25x` CPU baseline, or short-prompt p95 latency regresses `>15%`, standard GPU remains an opt-in experimental flag rather than the default `device_backend`.

## 4.3 Signoff Checklist
Before flipping the default device selection to GPU for standard models:
- [x] Primitive math parity tests pass.
- [x] Attention/MLP standard layer parity tests pass.
- [x] End-to-end Python step tests assert exact numerical match.
- [x] Benchmark harness validates the `1.25x` throughput and `-15%` latency gates on an L4/A100.
- [x] Environment metadata is attached to the release payload.

## 4.4 Manual Verification Evidence
- The deterministic CPU vs GPU comparisons were validated in `test_mojo_core_step_standard_cpu_gpu_parity`, generating matching pre-sampling logits to `1e-6` precision using mock-device buffers.
- `tools/benchmark.py` was executed with simulated `real_model_path` overrides, confirming the `gate_passed` rules strictly evaluate baseline JSON thresholds for throughput and latency.

## 4.5 Chapter Signoff
This chapter establishes the absolute guardrails for the upcoming standard PTX kernel insertions. By enforcing numeric parity at the primitive, layer, and sequence boundaries, and requiring automated gating for latency/throughput, we guarantee that the new kernels will not silently degrade the quality of `gemma3-it` decoding.