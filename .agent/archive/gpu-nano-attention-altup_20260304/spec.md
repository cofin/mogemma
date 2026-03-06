# Flow: gpu-nano-attention-altup_20260304

## Flow Context
- Flow ID: `gpu-nano-attention-altup_20260304`
- Status: `planned`
- Parent PRD: `mojo-gpu-nano-parity_20260304`
- Chapter: `1` (Nano attention + AltUp/Laurel/per-layer mapping GPU parity)
- Beads Epic: `mogemma-13d.1`

## Local Status Snapshot

1. Beads is currently unreliable in this workspace, so local spec artifacts are the current source for implementation status.
2. This flow remains unimplemented locally.
3. Upstream GPU device-selection Phase 2 is now implemented locally in `gpu-device-selection-api_20260304`, so backend/device-selection semantics no longer need to be rediscovered here.
4. No nano GPU kernel, AltUp, Laurel, or per-layer-mapping code changes have been applied for this flow yet.

## Specification
Port nano-specific attention and stream-mixing math (`AltUp`, `Laurel`, `per_layer_map`) onto GPU dispatch points without changing established nano semantics.

### Code Analysis Summary
Files examined:
- `src/mo/mogemma/layers.mojo`
- `src/mo/mogemma/core.mojo`
- `src/mo/mogemma/model.mojo`
- `src/mo/mogemma/ops.mojo`
- `src/py/mogemma/convert.py`

Key findings:
1. Nano attention already encodes critical semantic rules that GPU path must preserve: weighted RMSNorm for `q/k`, unit RMSNorm for `v`, RoPE application, optional KV writes, and attention scaling fixed to `1.0` in `forward_attention_nano` (`src/mo/mogemma/layers.mojo:404`).
2. AltUp prediction/correction is structurally coupled to router outputs and coefficient tensor layout assumptions; `forward_altup_predict` and `forward_altup_correct` rely on explicit modality loops that are sensitive to ordering (`src/mo/mogemma/layers.mojo:545`, `src/mo/mogemma/layers.mojo:584`).
3. Laurel and per-layer mapping participate directly in active/non-active stream updates, so GPU migration must preserve operation order (`forward_laurel`, `forward_per_layer_mapping`, `forward_nano_layer`) (`src/mo/mogemma/layers.mojo:505`, `src/mo/mogemma/layers.mojo:480`, `src/mo/mogemma/layers.mojo:728`).
4. Per-layer token inputs are built from `per_layer_projection` and `per_layer_embed` with runtime table depth (`shape_1`), not a hardcoded layer count (`src/mo/mogemma/layers.mojo:660`).
5. Runtime currently rebuilds `NanoModelWeights` on each token step via `_build_nano_model_from_runtime` inside `step_mojo`, so GPU integration must align with broader runtime-caching work (`src/mo/mogemma/core.mojo:201`, `src/mo/mogemma/core.mojo:366`).
6. Conversion layout validation guards tensor shapes for AltUp/per-layer mapping but does not validate runtime math parity; parity must be gated separately (`src/py/mogemma/convert.py:262`).

### Relevant Patterns
- Gemma3 nano uses weighted RMSNorm (`w * x`), dedicated `v_norm`, and attention scaling `1.0`; do not reuse standard RMSNorm semantics (`.agent/patterns.md`).
- AltUp dimensions must be sourced from `per_layer_map.gate.weight` contracts, not inferred from `per_layer_embed.shape_1` (`.agent/patterns.md`).
- Conversion-shape correctness is necessary but insufficient; runtime parity checks are required (`.agent/patterns.md`).
- Deterministic validation should use `temperature=0` and `top_k=1` to expose regressions (`.agent/patterns.md`).

### Primary Touch Points
- `src/mo/mogemma/core.mojo`: session/backend selection, runtime cache reuse, and nano dispatch entrypoints.
- `src/mo/mogemma/layers.mojo`: nano attention, AltUp predict/correct, Laurel, per-layer mapping, and stream update ordering.
- `src/mo/mogemma/ops.mojo` or a dedicated sibling GPU nano module: backend-specific kernels and scratch helpers.
- `src/mo/tests/test_altup_contract.mojo`, `src/mo/tests/test_nano_layers.mojo`, and `src/py/tests/test_mojo_core.py`: primitive, integration, and runtime-visible parity surfaces.

### Requirements
#### Functional Requirements
1. Introduce GPU-dispatchable nano kernels for attention, AltUp predict/correct, Laurel, and per-layer mapping with tensor layouts compatible with current `NanoLayerWeights` wiring.
2. Preserve operation order and math equivalence for:
- `forward_attention_nano`
- `forward_altup_predict`
- `forward_altup_correct`
- `forward_laurel`
- `forward_per_layer_mapping`
3. Keep CPU path as canonical fallback and ensure outputs remain equivalent within defined tolerances.
4. Ensure per-layer input construction still uses runtime layer-axis metadata (`per_layer_table_layers`) and validated per-layer dimensions.
5. Freeze contract units separately for nano attention, AltUp predict, AltUp correct, Laurel, and per-layer mapping. Each unit contract must name input/output shapes, dtype, scratch ownership, required precomputed tables, and parity target.
6. Freeze numeric parity classes before implementation:
- primitive/kernel parity: `atol <= 1e-6`
- integrated hidden/logit parity: `atol <= 1e-4`, `rtol <= 1e-4`
- deterministic decode: exact token parity
- `write_kv`, modality ordering, and `per_layer_table_layers` usage: exact-match contract
7. Dispatch injection must be session-latched at init/runtime setup, not re-resolved per call or per layer.

#### Non-Functional Requirements
1. No hidden fallback behavior: dispatch decisions must be explicit, traceable, and impossible to change after the first token of a sequence.
2. GPU path must avoid extra host-device synchronization inside per-layer loops beyond required correctness boundaries.
3. Nano hot path must perform zero model-descriptor rebuilds after init and no host round-trips for per-layer math.
4. GPU implementation must reuse scratch/device allocations across tokens and emit per-token sync/copy counters in debug or test mode.
5. Error messages for shape mismatches must remain fail-fast and actionable.

#### Risks
1. Modality ordering or coefficient indexing drift can silently degrade AltUp quality.
2. RMSNorm variant mix-ups (`1 + w` vs `w`) can pass shape checks but fail semantic parity.
3. Per-layer mapping/per-layer-input dimension mismatch can break late in runtime if not prevalidated.

## Implementation Plan
### Phase 1: Contracts and Dispatch Boundaries
- [x] 1.1 Define explicit kernel contracts for nano attention, AltUp predict, AltUp correct, Laurel, and per-layer mapping (inputs, outputs, strides, dtypes, scratch ownership, precomputed tables, parity target).
- [x] 1.2 Define the GPU dispatch seam in `core.mojo`/`layers.mojo` without changing public Python API, and make the backend choice session-latched rather than per-call.
- [x] 1.3 Checkpoint: produce an ownership/contract table mapping each CPU function to its GPU equivalent, touched module, and frozen parity tolerance.

### Phase 2: TDD Harness for Chapter 1 Math
- [x] 2.1 Add failing deterministic parity tests for `forward_attention_nano` (including `write_kv=True/False`).
- [x] 2.2 Add failing deterministic parity tests for AltUp predict and AltUp correct routing (shape + value checks, modality ordering, coefficient-axis expectations).
- [x] 2.3 Add failing deterministic parity tests for Laurel and per-layer mapping deltas as separate fixtures with explicit `per_layer_table_layers` coverage.
- [x] 2.4 Checkpoint: all new tests fail on stub GPU path and pass on CPU reference path.

### Phase 3: GPU Implementation and Integration
- [x] 3.1 Implement GPU kernels/dispatch for nano attention with preserved weighted RMSNorm and `v_norm` behavior.
- [x] 3.2 Implement GPU kernels/dispatch for AltUp predict/correct, Laurel, and per-layer mapping as individually testable deliverables, preserving modality ordering, coefficient indexing, stream update ordering, and runtime layer-axis usage.
- [x] 3.3 Integrate kernels into nano token hidden flow while preserving stream update ordering.
- [x] 3.4 Checkpoint: deterministic CPU/GPU parity tests pass for isolated kernels and integrated nano layer, with stable sync/copy counters and no post-init descriptor rebuilds.

### Phase 4: End-to-End Evidence
- [x] 4.1 Run nano end-to-end deterministic generation parity checks at fixed prompt set.
- [x] 4.2 Check in a tolerance table covering primitive parity, integrated parity, and deterministic token parity, with any accepted drift explicitly justified.
- [x] 4.3 Manual verification: run a local nano prompt smoke test in deterministic mode and confirm no gibberish regression versus CPU baseline.
- [x] 4.4 Checkpoint: chapter-ready evidence packet includes unit parity, integration parity, checked-in tolerance table, cold-vs-warm latency comparison, and manual verification notes.

## Performance Evidence
- Record per-token sync count and device scratch reuse behavior for the nano layer loop.
- Record cold-vs-warm latency on a fixed deterministic prompt to prove session-latched dispatch and runtime-cache reuse.
- Reject designs that require allocator churn or host copies inside the per-layer nano loop.

## TDD-Oriented Task Checklist
- [x] Red: add attention parity fixtures that fail when weighted RMSNorm or scale `1.0` is violated.
- [x] Red: add AltUp routing fixtures that fail on modality-index ordering drift.
- [x] Red: add per-layer mapping fixtures that fail when `per_layer_dim` contract is broken.
- [x] Green: implement GPU dispatch until all chapter-1 red tests pass.
- [x] Refactor: reduce duplication between CPU and GPU path setup while keeping test green.
- [x] Regression: rerun deterministic nano prompt and embedding sanity checks after refactor.
