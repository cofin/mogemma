# Research: GPU acceleration options for `mogemma`

Date: 2026-03-04
Type: Performance / Architecture
Topic: Identify practical GPU acceleration paths for current `mogemma` codebase.

## Executive Summary

`mogemma` is currently CPU-only in practice. The `device` field exists in config but is not wired to execution paths, and core inference is implemented with CPU-oriented loops in Mojo.

You have four realistic GPU paths:

1. **MAX Engine backend (highest performance + fastest time-to-value, but license/process implications).**
2. **Pure Mojo GPU kernels via `gpu.host.DeviceContext` (no MAX runtime in serving path, highest control, largest engineering effort).**
3. **Hybrid pure-Mojo + vendor library FFI (cuBLAS/ROCm) for GEMM-heavy hotspots (non-MAX, medium effort, strong perf potential).**
4. **PyTorch runtime + Mojo custom kernels via `max.torch` (good bridge option, but still depends on Modular/MAX toolchain).**

Given your explicit constraint around MAX licensing implications, the best primary direction is:
- **Non-MAX track:** implement a pure-Mojo GPU execution path first (Option 2), then selectively add FFI to vendor libraries for bottlenecks (Option 3).

## Codebase Analysis

### Relevant files

- `src/py/mogemma/config.py`
- `src/py/mogemma/model.py`
- `src/mo/mogemma/core.mojo`
- `src/mo/mogemma/layers.mojo`
- `src/mo/mogemma/ops.mojo`
- `pyproject.toml`
- `.agent/archive/pure-mojo-bridge/spec.md`
- `.agent/knowledge/pure-mojo-bridge.md`

### Current execution constraints

1. **`device` is effectively a no-op.**
- `EmbeddingConfig.device` and `GenerationConfig.device` exist, but runtime path does not read/use them.

2. **Inference core is CPU-style memory and kernels.**
- `step_mojo()` allocates NumPy logits and Mojo scratch each token.
- `forward_*` and `ops.mojo` use CPU SIMD loops over `UnsafePointer`, not GPU launch APIs.

3. **Per-token rebuild overhead exists before any GPU work.**
- `step_mojo()` rebuilds model-weight wrappers per token (`_build_model_from_runtime` / `_build_nano_model_from_runtime`), which is a measurable overhead even on CPU.

4. **Build/runtime currently compiles Python extension only.**
- Wheel hook emits `mogemma._core` Python extension; no dedicated GPU backend module or runtime selector exists.

### Implication

Before full GPU acceleration, `mogemma` should separate:
- runtime/session state
- model graph/weights metadata
- device-specific kernel dispatch

That architecture change is necessary for all four options.

## Library Documentation (Current Capabilities)

### Mojo-native GPU programming (non-MAX viable)

Mojo docs now explicitly describe GPU programming with:
- `gpu.host.DeviceContext`
- kernel compilation/launch (`compile_function`, `enqueue_function`)
- vendor API selection (`api="cuda"|"hip"|"metal"`)
- accelerator detection via `sys.has_*_gpu_accelerator()`

This confirms a viable non-MAX GPU path for Mojo code.

### MAX capabilities (reference path)

MAX docs show:
- model serving on CPU/GPU with device selection (`--devices=cpu|gpu|gpu:0,1`)
- optimized model graphs and model support pipeline
- custom ops in Mojo for GPU and CPU
- `max.engine.InferenceSession` with device lists and GPU profiling

This is still the shortest path to highest throughput for large-model serving.

### Hardware support snapshot (from current docs)

Current docs list tiered GPU support across NVIDIA, AMD, Apple silicon, plus driver requirements. This is relevant for planning portability and CI matrix.

### Licensing context

Modular Community License terms apply to software developed with MAX/Mojo SDKs and include usage/distribution constraints and hardware/distribution clauses. This is the core reason to treat MAX-coupled vs non-MAX strategies separately.

## Prior Art

### In-repo prior art

1. `pure-mojo-bridge` intentionally removed MAX/modular dependencies to reduce footprint and complexity.
2. Historical plan noted benchmark comparisons against native MAX execution.
3. Current architecture and tests assume `_core` as a Python extension boundary; this is a stable seam for introducing backend dispatch.

### External prior art

1. MAX custom ops tutorial: GPU/CPU custom ops in Mojo with graph integration.
2. MAX + PyTorch interop (`max.torch.CustomOpLibrary`): targeted acceleration without full graph migration.
3. Mojo GPU fundamentals: standalone GPU programming model with explicit memory transfer and synchronization.

## Option Matrix

### Option 1: Reintroduce MAX backend for generation

- Description: add optional backend using `max serve`/`max.engine` for inference while preserving `mogemma` API.
- Perf ceiling: highest near-term.
- Engineering cost: low to medium.
- Risks: licensing/process implications, dependency footprint increase, drift from pure-mojo architecture.
- Fit with your constraint: weak.

### Option 2: Pure Mojo GPU backend using `gpu.host` APIs

- Description: implement GPU kernels for core ops (matmul/attention/MLP/norm), device buffers, and async launch in `mogemma._core`.
- Perf ceiling: high, especially with tuned kernels.
- Engineering cost: high.
- Risks: correctness complexity (especially nano path), kernel tuning burden, multi-vendor maintenance.
- Fit with your constraint: strong.

### Option 3: Pure Mojo + FFI to vendor GPU libraries

- Description: keep non-MAX runtime, call cuBLAS/ROCm equivalents from Mojo via FFI for GEMM-heavy blocks.
- Perf ceiling: high for GEMM-bound sections; attention path still needs dedicated kernels.
- Engineering cost: medium-high.
- Risks: platform-specific ABI handling, packaging complexity, limited portability.
- Fit with your constraint: strong.

### Option 4: PyTorch runtime + Mojo custom ops (`max.torch`)

- Description: run model in PyTorch, replace hotspots with Mojo kernels via custom op library.
- Perf ceiling: medium-high depending on partitioning.
- Engineering cost: medium.
- Risks: dual runtime complexity, still uses MAX tooling, conversion overhead and operational complexity.
- Fit with your constraint: medium/weak.

## Risk Assessment

1. **Architecture risk (high):** GPU acceleration without refactoring token-step state and model wrapper caching will underperform.
2. **Correctness risk (high):** Nano attention/AltUp/per-layer mapping already has strict parity requirements; GPU rewrites increase regression risk.
3. **Packaging risk (medium-high):** shipping GPU dependencies in wheels for multiple CUDA/ROCm environments is non-trivial.
4. **License/compliance risk (contextual):** MAX-coupled paths require legal review against your deployment model.
5. **Supportability risk (medium):** multi-vendor GPU matrix (NVIDIA/AMD/Apple) increases CI and QA burden.

## Recommended Approach

### Recommendation under your constraint

Adopt a **non-MAX staged plan**:

1. **Stage A (foundation):** Refactor current core for backend pluggability and remove per-token model-wrapper rebuild.
2. **Stage B (GPU MVP):** Add CUDA-first pure-Mojo path using `DeviceContext`, starting with standard Gemma path.
3. **Stage C (throughput):** Offload GEMM hotspots via FFI to vendor libraries where beneficial.
4. **Stage D (coverage):** add AMD path and then nano-path GPU parity.

### Why this is best

- Preserves pure-mojo architecture objective.
- Avoids immediate MAX coupling/licensing complications.
- Keeps highest long-term control over distribution and runtime behavior.
- Allows incremental performance wins without a full rewrite in one step.

### Fastest fallback if timelines dominate

If time-to-performance matters more than license/process concerns, Option 1 (MAX backend) is the fastest way to ship GPU acceleration.

## Suggested PRD Inputs

1. Add backend interface (`cpu_mojo`, `gpu_mojo`, optional `max_backend`).
2. Define device selection API (`device="cpu|gpu|gpu:0"`) and failure policy.
3. Establish parity gates:
- token-level deterministic fixtures
- semantic output tests
- perf thresholds (tokens/sec) per backend.
4. Define supported GPU matrix (initially NVIDIA CUDA only).
5. Define packaging/deployment strategy for GPU builds.

## Sources

### Local sources

- `src/py/mogemma/config.py`
- `src/py/mogemma/model.py`
- `src/mo/mogemma/core.mojo`
- `src/mo/mogemma/layers.mojo`
- `src/mo/mogemma/ops.mojo`
- `pyproject.toml`
- `.agent/archive/pure-mojo-bridge/spec.md`
- `.agent/knowledge/pure-mojo-bridge.md`
- `.agent/archive/text-inference-engine/plan.md`

### External sources

- Mojo GPU fundamentals:
  - https://docs.modular.com/mojo/manual/gpu/fundamentals/
- Mojo `sys` accelerator detection:
  - https://docs.modular.com/mojo/std/sys/info/
- Mojo `gpu.host` APIs:
  - https://docs.modular.com/mojo/std/gpu/host/
  - https://docs.modular.com/mojo/std/gpu/host/device_context/DeviceContext/
- Mojo FFI:
  - https://docs.modular.com/mojo/std/ffi/
- MAX model/custom-op ecosystem:
  - https://docs.modular.com/max/develop/custom-ops/
  - https://docs.modular.com/max/api/python/torch/
  - https://docs.modular.com/max/develop/custom-kernels-pytorch/
  - https://docs.modular.com/max/model-formats/
  - https://docs.modular.com/max/api/python/engine/
  - https://docs.modular.com/max/cli/serve/
- Current package/system/GPU support matrix:
  - https://docs.modular.com/max/packages/
  - https://docs.modular.com/mojo/manual/install/
- Licensing terms:
  - https://www.modular.com/legal/community
