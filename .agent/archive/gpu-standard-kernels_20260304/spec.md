# Flow: gpu-standard-kernels_20260304

**Flow ID:** `gpu-standard-kernels_20260304`  
**Status:** `Planned`  
**Parent Context:** `mojo-gpu-standard-kernels_20260304` (Chapter 1: GPU kernelization for standard Gemma path)  
**Beads Epic:** `mogemma-3hl.1`

## Local Status Snapshot

1. Beads is currently unreliable in this workspace, so local spec artifacts are the current source for implementation status.
2. This flow remains unimplemented locally.
3. Upstream GPU device-selection Phase 2 is now implemented locally in `gpu-device-selection-api_20260304`, so standard-kernel work can assume normalized device/backend policy at the Python boundary.
4. No standard GPU-kernel or `_core.step` dispatch changes have been applied for this flow yet.

## Specification

### Parent Context

PRD `mojo-gpu-standard-kernels_20260304` defines this chapter as the first delivery slice: add CUDA-first acceleration for standard Gemma attention/MLP/norm/RoPE kernels while preserving CPU behavior and preparing clean hand-off to device-memory and correctness-gate chapters.

### Code Analysis Summary

**Files Examined**

- `src/mo/mogemma/core.mojo` - runtime assembly, standard/nano branch split, and `step_mojo` entrypoint.
- `src/mo/mogemma/layers.mojo` - standard attention, MLP, layer, and token-step orchestration.
- `src/mo/mogemma/ops.mojo` - scalar/SIMD CPU primitives used by standard path.
- `src/mo/mogemma/model.mojo` - standard tensor and layer structs used by kernel callers.
- `src/py/mogemma/model.py` - Python generation path invoking backend `_core.step`.
- `src/py/mogemma/backends.py` - resolved backend/device contract already exposed at Python boundary.
- `src/mo/tests/test_ops.mojo` - primitive correctness expectations.
- `src/mo/tests/test_layers.mojo` - standard layer and attention numerical fixtures.
- `src/py/tests/test_mojo_core.py` - standard `_core.init_model()` and `_core.step()` shape/state checks.

**Key Findings**

- Standard generation calls `_forward_step_standard_runtime(...)` through `step_mojo` in the non-nano branch, so `core.mojo` and `layers.mojo` are the mandatory integration seam for GPU dispatch (`src/mo/mogemma/core.mojo:955`, `src/mo/mogemma/core.mojo:991`, `src/mo/mogemma/layers.mojo:224`).
- Runtime-state refactor work already removed per-token model rebuilding, so this flow must not reintroduce descriptor/model construction in the hot path (`src/mo/mogemma/core.mojo:913`, `src/mo/mogemma/core.mojo:955`).
- Standard attention is still a pure pointer-loop pipeline: Q/K/V projections, optional QK norm, RoPE, KV write, score/softmax/value reduce, output projection (`src/mo/mogemma/layers.mojo:15`, `src/mo/mogemma/layers.mojo:32`, `src/mo/mogemma/layers.mojo:46`, `src/mo/mogemma/layers.mojo:82`, `src/mo/mogemma/layers.mojo:97`).
- Primitive ops are CPU vectorized loops, not backend-dispatched kernels, so Chapter 1 needs an explicit standard-path GPU execution surface rather than deeper branching in Python (`src/mo/mogemma/ops.mojo:5`, `src/mo/mogemma/ops.mojo:62`, `src/mo/mogemma/ops.mojo:92`, `src/mo/mogemma/ops.mojo:127`).
- Python already resolves device/backend policy before calling `_core`, so this flow must consume existing backend/device signals instead of redefining public API semantics (`src/py/mogemma/model.py:405`, `src/py/mogemma/backends.py:139`).
- Existing tests validate CPU math and tensor contracts, but there is no standard GPU execution-path coverage yet (`src/mo/tests/test_ops.mojo:1`, `src/mo/tests/test_layers.mojo:1`, `src/py/tests/test_mojo_core.py:18`).

### Relevant Patterns

- **Hybrid Mojo-Python boundary:** keep `_core` as the Python extension boundary; kernel/backend dispatch stays behind this interface.
- **Deterministic validation:** use `temperature=0` and `top_k=1` for parity and correctness checks.
- **Model rebuild overhead warning:** do not add new per-token initialization or backend re-resolution work.
- **Standard-only scope:** this chapter must not change nano execution semantics or ownership boundaries.

### Implementation Guardrails

**Primary Touch Points**

- `src/mo/mogemma/core.mojo` - select and latch standard-path backend execution once per session; expose debug counters and deterministic fallback behavior.
- `src/mo/mogemma/layers.mojo` - add backend-dispatched standard-path entrypoints for attention, layer, and step orchestration.
- `src/mo/mogemma/ops.mojo` or a dedicated sibling GPU-kernel module under `src/mo/mogemma/` - hold backend-specific primitive implementations without polluting Python-facing contracts.
- `src/mo/tests/test_ops.mojo`, `src/mo/tests/test_layers.mojo`, and `src/py/tests/test_mojo_core.py` - prove parity, fallback, and standard-path routing.

**Non-Goals**

- No public Python API changes in `src/py/mogemma/model.py` or config grammar changes in this chapter.
- No nano-path kernelization, no KV-share changes, and no embedding-path redesign beyond what is necessary to preserve shared core contracts.
- No mixed-backend session behavior; a session must not silently swap between CPU and GPU mid-run.

**Hot-Path Invariants**

- Preserve standard RMSNorm semantics `(1 + weight)`, RoPE ordering, KV write-before-attend ordering, and 1D float32 logits output contract.
- Backend selection must be resolved once per standard session and reused for every token; no per-token backend lookup.
- Chapter 1 may upload token input and download logits per token, but must not introduce per-layer host-device copies or per-head sync points.
- Kernel signatures must already be compatible with Chapter 2 device-resident KV/scratch buffers so Chapter 2 can change buffer ownership without changing compute interfaces.

### Requirements

#### Functional Requirements

- FR1. Add a standard-path GPU kernel/dispatch design covering attention, MLP, RMSNorm, RoPE, and softmax without changing external Python API.
- FR2. Preserve CPU execution path as explicit fallback and baseline.
- FR3. Introduce clear standard-path backend selection points in Mojo core so later chapters can attach device-memory and gate logic.
- FR4. Keep standard numerical semantics unchanged, including current `(1 + weight)` RMSNorm behavior and existing KV cache indexing.
- FR5. Keep nano path out of scope for implementation changes in this flow.
- FR6. Define session-level fallback latching so capability/init failures before generation begins can fall back once, while mid-session kernel failures abort deterministically instead of mixing caches/backends.

#### Non-Functional Requirements

- NFR1. Plan must identify hot-path performance risks and forbid new per-token rebuild or backend-resolution overhead.
- NFR2. Standard GPU dispatch must avoid per-layer host-device copies and per-head synchronization in the token loop.
- NFR3. Plan must define test hooks for unit-level kernel checks, layer fixtures, and end-to-end token-step checks.
- NFR4. Rollout must stay incremental and debuggable, with backend/fallback state visible through deterministic counters or logs.

#### Risks

- R1. Kernel parity drift in QK norm, RoPE, or softmax ordering can silently break output quality.
- R2. Too-fine launch granularity can make GPU path slower than CPU despite correct math.
- R3. Dispatch complexity in `step_mojo` can regress CPU baseline latency if fallback/selection is resolved per token.
- R4. Over-coupling with nano code paths can introduce unintended behavioral regressions.

## Implementation Plan

### Phase 1: Standard Kernel Scope and Dispatch Contract

- [x] 1.1 Map the exact standard call graph `step_mojo -> _forward_step_standard_runtime -> forward_step -> forward_layer -> forward_attention/forward_mlp -> vec_mat_mul|rms_norm|rope_rotate|softmax|geglu`, with explicit non-goals for nano and embedding-only paths.
- [x] 1.2 Define ownership boundaries: `core.mojo` owns session backend latch and fallback policy, `layers.mojo` owns standard graph orchestration, and `ops.mojo` or a dedicated GPU sibling module owns backend-specific kernels.
- [x] 1.3 Define kernel-entry invariants: contiguous float32 pointers, scratch-layout expectations, cache pointer semantics, logits shape, and explicit prohibition on nano-only fields.
- [x] 1.4 Checkpoint: freeze a dispatch/invariants table that names every standard symbol that remains CPU-only versus GPU-dispatched in this chapter.

### Phase 2: Kernelization Design for Standard Primitives

- [x] 2.1 Define grouped/fused GPU kernel coverage for Q/K/V projection, optional Q/K norm, RoPE, score softmax, value accumulation, output projection, RMSNorm, and GEGLU.
- [x] 2.2 Define the scratch-memory contract so attention/layer kernels consume one contiguous scratch region compatible with future device-resident scratch pools from Chapter 2.
- [x] 2.3 Define the standard MLP execution plan, including where projection fusion is allowed and where CPU fallback can safely cut over.
- [x] 2.4 Checkpoint: validate that every operation reached from `forward_step` has an explicit GPU implementation path or an explicit CPU-only/fallback rationale.

### Phase 3: Core Integration and Fallback Behavior

- [x] 3.1 Integrate a standard-only backend latch in `step_mojo` so GPU vs CPU dispatch is selected once per session and reused for every token.
- [x] 3.2 Define fallback semantics precisely: capability/init failures before first token may latch CPU for the session; failures after cache mutation must raise deterministic errors and stop the session.
- [x] 3.3 Add deterministic debug surfaces for active backend, fallback reason, and per-token launch/copy counters (debug/test mode only is acceptable).
- [x] 3.4 Checkpoint: prove CPU fallback correctness, no nano-path coupling, and no per-token backend re-resolution.

### Phase 4: Validation Design and Handoff to Gate Chapter

- [x] 4.1 Add Mojo unit coverage for primitive parity in `src/mo/tests/test_ops.mojo` and standard attention/layer parity in `src/mo/tests/test_layers.mojo`.
- [x] 4.2 Add Python integration coverage in `src/py/tests/test_mojo_core.py` for standard `_core.step` routing, fallback latching, and deterministic logits/argmax parity.
- [x] 4.3 Record performance evidence required before default enablement: GPU medium/long-prompt throughput must beat CPU on the designated GPU host, and short-prompt p95 latency regression must stay within the agreed bound or the path remains opt-in.
- [x] 4.4 Manual verification: run deterministic prompt smoke tests on CPU fallback and GPU standard dispatch, compare token outputs/logit deltas, and record backend/fallback counters.
- [x] 4.5 Checkpoint: implementation-ready signoff includes exact touched files, fallback policy, kernel coverage map, and gate/evidence handoff for Chapter 3.

## TDD-Oriented Task Checklist

- [x] RED: add failing primitive parity tests in `src/mo/tests/test_ops.mojo` for standard GPU-dispatched `vec_mat_mul`, `rms_norm`, `rope_rotate`, `softmax`, and `geglu`.
- [x] RED: add failing layer/step parity tests in `src/mo/tests/test_layers.mojo` and `src/py/tests/test_mojo_core.py` that compare CPU vs GPU standard-path behavior under deterministic settings.
- [x] GREEN: introduce standard GPU kernel scaffolding plus session-latched backend selection without changing Python API contracts.
- [x] GREEN: implement explicit fallback behavior and pass both GPU-present and GPU-absent test runs.
- [x] REFACTOR: consolidate scratch-layout packing and backend-call plumbing so no per-token backend lookup or duplicate argument marshalling remains.
- [x] REGRESSION: run `src/mo/tests/test_ops.mojo`, `src/mo/tests/test_layers.mojo`, and targeted `src/py/tests/test_mojo_core.py`.
