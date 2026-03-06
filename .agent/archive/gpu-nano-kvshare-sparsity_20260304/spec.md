# Flow: gpu-nano-kvshare-sparsity_20260304

## Flow Context
- Flow ID: `gpu-nano-kvshare-sparsity_20260304`
- Status: `planned`
- Parent PRD: `mojo-gpu-nano-parity_20260304`
- Chapter: `2` (KV sharing + activation sparsity GPU parity)
- Beads Epic: `mogemma-13d.2`

## Local Status Snapshot

1. Beads is currently unreliable in this workspace, so local spec artifacts are the current source for implementation status.
2. This flow remains unimplemented locally.
3. Upstream GPU device-selection Phase 2 is now implemented locally in `gpu-device-selection-api_20260304`, so backend/device-selection semantics are fixed input context for this chapter.
4. No KV-sharing or sparsity GPU-path changes have been applied for this flow yet.

## Specification
Implement GPU-aware KV-sharing and activation-sparsity behavior for nano layers while preserving existing CPU semantics.

### Code Analysis Summary
Files examined:
- `src/mo/mogemma/core.mojo`
- `src/mo/mogemma/layers.mojo`
- `src/mo/mogemma/model.mojo`
- `src/mo/mogemma/ops.mojo`
- `src/py/mogemma/convert.py`

Key findings:
1. KV-sharing start is inferred at init by scanning for first post-nonzero layer with effectively-zero `k_proj` and `v_proj` (`_detect_nano_kv_share_start`) (`src/mo/mogemma/core.mojo:261`).
2. Nano token forward uses explicit layer remapping for shared KV (`kv_layer_idx`) and suppresses writes (`write_kv=False`) after `kv_share_start`, with full-attention-vs-sliding behavior encoded as every 5th layer rule (`src/mo/mogemma/layers.mojo:896`).
3. `forward_attention_nano` branches on `write_kv` and reads from remapped cache even when projections are skipped; correctness depends on consistent cache indexing (`src/mo/mogemma/layers.mojo:435`).
4. Nano activation sparsity is implemented as Gaussian cutoff masking over gate activations (z-score for 0.95) and applied only for first 10 layers (`src/mo/mogemma/layers.mojo:121`, `src/mo/mogemma/layers.mojo:150`).
5. Kernel primitives are CPU SIMD loops (`vec_mat_mul`, `softmax`, `geglu`), so GPU path must replicate exact thresholding and cache semantics explicitly (`src/mo/mogemma/ops.mojo:69`).

### Relevant Patterns
- KV sharing in `gemma3n-e2b-it` (layers 20..29 zeroed `k_proj`/`v_proj`) is a required behavior, not an optimization (`.agent/patterns.md`).
- Early-layer nano activation sparsity (default `0.95`) is quality-critical; removing/skipping it degrades outputs (`.agent/patterns.md`).
- Deterministic validation settings are required for parity checks (`temperature=0`, `top_k=1`) (`.agent/patterns.md`).

### Primary Touch Points
- `src/mo/mogemma/core.mojo`: `kv_share_start` detection, runtime/session fields, and reset semantics.
- `src/mo/mogemma/layers.mojo`: `kv_layer_idx` remapping, `write_kv` suppression, and sparsity mask generation.
- `src/mo/mogemma/model.mojo`: cache/device-handle layout carried through nano runtime state.
- `src/mo/tests/test_nano_layers.mojo` and `src/py/tests/test_mojo_core.py`: cache remap, reset/reuse, and long-sequence parity coverage.

### Cache ABI Contract
1. `kv_share_start` is computed once at init from layer tensors and then treated as immutable runtime metadata.
2. `kv_layer_idx` must follow an explicit truth table for pre-share, shared-full, and shared-sliding layers.
3. Every 5th shared layer remains full attention; all other shared layers retain sliding behavior.
4. `write_kv` suppression for shared layers is an exact-match contract, not a tolerance-based behavior.
5. Device cache handles, reset behavior, and ownership/freeing rules must be explicit and test-visible.

### Requirements
#### Functional Requirements
1. Preserve `_detect_nano_kv_share_start` semantics and expose equivalent behavior for GPU runtime initialization.
2. Preserve nano KV layer remapping logic for shared layers, including full-attention/sliding split and `write_kv` suppression behavior.
3. Preserve activation sparsity semantics in nano MLP:
- z-score thresholding behavior at sparsity `0.95`
- layer gating limited to first 10 decoder layers
4. Ensure parity across both `forward_nano_step` and `forward_nano_sequence` code paths.
5. Freeze exact truth tables for `kv_share_start`, `kv_layer_idx`, `write_kv`, and sparse-vs-dense layer gating before implementation begins.

#### Non-Functional Requirements
1. KV-sharing detection must happen once per initialized runtime, not per token.
2. GPU kernels must avoid race conditions when reading shared KV cache regions.
3. Numerical tolerance policy for sparsity outputs must be explicit and documented.
4. Zero steady-state cache/scratch reallocations are allowed once generation begins; any extra allocation must fail a debug/test counter.
5. Any temporary host mirrors must remain bounded and debug-only; steady-state duplication above 10% of required session bytes requires written rationale.
6. Logits egress is allowed at most one explicit sync per token step, and reset behavior must be deterministic.

#### Risks
1. Off-by-one errors in `kv_layer_idx` remapping can produce coherent-but-wrong outputs.
2. Parallelized mean/std computation for sparsity can alter threshold behavior if reduction order is unstable.
3. Shared KV cache aliasing bugs can introduce cross-layer contamination that is hard to diagnose.

## Implementation Plan
### Phase 1: KV-Share and Sparsity Contracts
- [x] 1.1 Document expected `kv_share_start` detection behavior with fixtures for zero/nonzero boundary cases and runtime caching semantics.
- [x] 1.2 Document the exact layer-remapping/write-suppression truth table for pre-share, shared-full, and shared-sliding layers.
- [x] 1.3 Document activation sparsity contract, including first-10-layer gating, mean/std reduction behavior, and explicit tolerance bounds for masks/gate outputs.
- [x] 1.4 Checkpoint: approved CPU-reference contract spec for KV-sharing and sparsity.

### Phase 2: TDD Coverage (Red First)
- [x] 2.1 Add failing tests for `_detect_nano_kv_share_start` against synthetic layer tensors.
- [x] 2.2 Add failing tests for layer-to-KV remapping decisions across boundary layers.
- [x] 2.3 Add failing tests for activation sparsity outputs (first 10 layers sparse, later layers dense).
- [x] 2.4 Add failing integration tests for shared-KV behavior in token-by-token generation, reset/reuse correctness, and long-sequence boundary prompts.
- [x] 2.5 Checkpoint: red tests isolate KV-sharing and sparsity regressions before GPU implementation.

### Phase 3: GPU Path Implementation
- [x] 3.1 Implement the shared-cache read path and suppressed-write path with exact remapping rules and deterministic device-handle ownership.
- [x] 3.2 Implement deterministic sparsity reduction, cutoff, and mask generation for nano MLP gate values with explicit layer gating.
- [x] 3.3 Integrate both behaviors into nano forward dispatch.
- [x] 3.4 Checkpoint: all KV-sharing + sparsity parity tests pass against CPU baseline with zero steady-state reallocations and stable copy/sync counters.

### Phase 4: Validation and Manual Verification
- [x] 4.1 Run deterministic nano prompt suite with KV-sharing enabled and compare against CPU output quality.
- [x] 4.2 Run targeted stress prompts covering sequence positions around KV-share boundaries, long-sequence cache aliasing, and reset/reuse transitions.
- [x] 4.3 Manual verification: inspect debug traces for representative layers to confirm `write_kv`, `kv_layer_idx`, sync/copy counts, and reset behavior match contract.
- [x] 4.4 Checkpoint: chapter-ready report includes parity pass summary, boundary-case evidence, and copy-count/perf evidence.

## TDD-Oriented Task Checklist
- [x] Red: write KV-share detection tests that fail for current GPU-stub path.
- [x] Red: write KV remap table tests for all layer classes (pre-share, shared-full, shared-sliding).
- [x] Red: write sparsity cutoff tests that fail if threshold math or layer gating drifts.
- [x] Green: implement GPU path until all KV-sharing and sparsity tests pass.
- [x] Refactor: centralize shared contract helpers used by step and sequence paths.
- [x] Regression: rerun deterministic nano generation sanity prompts after refactor.
