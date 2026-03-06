# Flow: gpu-kv-cache-device-memory_20260304

**Flow ID:** `gpu-kv-cache-device-memory_20260304`  
**Status:** `Planned`  
**Parent Context:** `mojo-gpu-standard-kernels_20260304` (Chapter 2: move standard-path KV/scratch memory to device)  
**Beads Epic:** `mogemma-3hl.2`

## Local Status Snapshot

1. Beads is currently unreliable in this workspace, so local spec artifacts are the current source for implementation status.
2. This flow remains unimplemented locally.
3. Upstream GPU device-selection Phase 2 is now implemented locally in `gpu-device-selection-api_20260304`, so this chapter can assume normalized device grammar/runtime descriptor semantics are available on the Python side.
4. No KV-cache or device-memory ownership changes have been applied for this flow yet.

## Specification

### Parent Context

PRD chapter 2 requires device-resident memory management for standard Gemma inference after kernelization: KV cache and critical per-step buffers must move to device with explicit transfer/synchronization policy, while CPU fallback remains supported and deterministic.

### Code Analysis Summary

**Files Examined**

- `src/mo/mogemma/core.mojo` - cache allocation, pointer extraction, runtime/session dictionary, and per-step scratch/logits allocation.
- `src/mo/mogemma/layers.mojo` - standard KV write/read behavior and per-layer cache indexing.
- `src/mo/mogemma/model.mojo` - `KVCache` struct and tensor container definitions.
- `src/py/mogemma/model.py` - caller behavior across token prefill/generation loops and session reset semantics.
- `src/py/mogemma/backends.py` - backend/device contract already exposed to Python.
- `src/py/tests/test_mojo_core.py` - current `_core` init/step assumptions around runtime dictionary and `pos`.
- `src/mo/tests/test_layers.mojo` - current KV cache assumptions in standard layer tests.

**Key Findings**

- `init_model_mojo` still allocates host `numpy` KV and RoPE buffers and stores them in the runtime dict (`src/mo/mogemma/core.mojo:874`, `src/mo/mogemma/core.mojo:879`, `src/mo/mogemma/core.mojo:894`).
- `step_mojo` still re-derives raw pointers from Python-owned arrays on every call and allocates transient scratch/logits buffers per token (`src/mo/mogemma/core.mojo:949`, `src/mo/mogemma/core.mojo:955`, `src/mo/mogemma/core.mojo:959`).
- Standard attention writes K/V per token to cache and later reads history using layer-major offsets, so any device migration must preserve exact indexing and write ordering (`src/mo/mogemma/layers.mojo:53`, `src/mo/mogemma/layers.mojo:58`, `src/mo/mogemma/layers.mojo:75`, `src/mo/mogemma/layers.mojo:255`).
- Session reset in Python currently assumes cache containers are zeroable/resettable from the wrapper (`src/py/mogemma/model.py:247`), so a device-resident design must define how reset remains deterministic.
- `generate_embeddings_mojo` still uses transient KV buffers sized to the input batch/sequence, which is a separate path and must not accidentally regress generation-session reuse (`src/mo/mogemma/core.mojo:1048`, `src/mo/mogemma/core.mojo:1071`).
- Public device/backend policy now lives in Python and should not be reinvented in this chapter (`src/py/mogemma/model.py:405`, `src/py/mogemma/backends.py:139`).

### Relevant Patterns

- **Hybrid Mojo-Python boundary:** preserve `_core` Python interface while changing internal memory ownership.
- **Deterministic validation:** use deterministic generation settings for parity checks during memory migration.
- **Model rebuild overhead warning:** device migration must remove host churn from the token loop, not merely move data structures.
- **CPU path preservation:** CPU session behavior and reset semantics remain the reference baseline.

### Implementation Guardrails

**Primary Touch Points**

- `src/mo/mogemma/core.mojo` - replace host-owned standard-session buffers with backend-owned handles/pools and thread them through `step_mojo`.
- `src/mo/mogemma/model.mojo` - define or extend stable buffer/session descriptors, including explicit KV/scratch ownership.
- `src/mo/mogemma/layers.mojo` - consume device-resident cache/scratch handles without changing standard indexing semantics.
- `src/py/tests/test_mojo_core.py` and `src/mo/tests/test_layers.mojo` - verify cache lifecycle, stable pointer/handle reuse, and long-sequence behavior.

**Non-Goals**

- No silent CPU/GPU migration after a GPU session has begun mutating cache state.
- No redesign of Python device grammar or backend-resolution rules in this chapter.
- No nano-path cache model changes; this chapter is standard-path only.

**Memory and Sync Invariants**

- Standard GPU sessions must allocate KV cache and reusable scratch once per session; zero cache or scratch reallocations are allowed in the steady-state token loop.
- Generation-step sync budget is at most one explicit sync before logits egress; there must be no per-layer or per-head host waits.
- Device path must preserve layer-major KV layout and `pos`/reset semantics so deterministic session reuse stays intact.
- Any temporary host mirrors are for fallback/debug only and must be explicitly bounded; steady-state duplication above 10% of required session bytes is not allowed without written rationale.

### Requirements

#### Functional Requirements

- FR1. Move standard-path KV cache to device-resident buffers with stable lifecycle across token steps.
- FR2. Define explicit host-device transfer policy for token input, logits output, and any required synchronization points.
- FR3. Move or pool critical per-step temporary buffers so standard GPU path avoids repeated host allocations.
- FR4. Preserve existing runtime/session contract expected by Python wrapper, or define a backward-compatible adapter with explicit reset behavior.
- FR5. Keep CPU path behavior and correctness intact as fallback.
- FR6. Define deterministic failure semantics for allocation/init errors before the first token and for runtime failures after cache mutation has started.

#### Non-Functional Requirements

- NFR1. Device memory plan must define exact steady-state byte formulas for KV, RoPE, scratch, and any host-visible staging buffers.
- NFR2. GPU step path must have zero steady-state host allocations and zero per-token cache pointer extraction from Python arrays.
- NFR3. Transfer policy must minimize PCIe/host copy overhead in token generation loop and record copy/sync counters for validation.
- NFR4. Reset/teardown behavior must be deterministic and leak-free across repeated generations.

#### Risks

- R1. Incorrect synchronization can produce stale KV reads and nondeterministic output drift.
- R2. Partial migration, such as moving caches but leaving scratch/logits on the host path, may erase performance wins.
- R3. Runtime ownership mismatch between Python-visible session state and device buffers can cause leaks or invalid pointers.
- R4. Embedding-path changes can accidentally regress current generation-session reuse if transient and session buffers are conflated.

## Implementation Plan

### Phase 1: Device Memory Contract for Standard Runtime

- [x] 1.1 Define the standard session-memory model: long-lived device KV buffers, long-lived step scratch, optional embedding scratch pools, and any host-visible staging buffers.
- [x] 1.2 Define which fields remain visible in `llm` for reset/debug (`pos`, `session_kv_cache_len`, `step_scratch_len`, `embedding_scratch_len`, active backend/debug counters) versus which stay opaque backend handles.
- [x] 1.3 Define the exact transfer/sync matrix for generation: token input upload, cache residency, logits egress, fallback/error path, and reset/teardown.
- [x] 1.4 Checkpoint: approve a lifecycle diagram covering init, steady-state step, reset, teardown, and error handling.

### Phase 2: KV Cache Device Residency

- [x] 2.1 Replace standard-session host `np.zeros` KV allocation with backend-owned device allocation while keeping CPU sessions unchanged.
- [x] 2.2 Thread cache handles/pointers through `_forward_step_standard_runtime` and `forward_attention` without changing layer-major indexing or write-before-attend semantics.
- [x] 2.3 Define fallback semantics precisely: device allocation failures before generation starts may create a CPU session instead; failures after cache mutation must raise and terminate the session.
- [x] 2.4 Checkpoint: verify every standard-path KV read/write path, reset path, and `pos` transition is covered by the new contract.

### Phase 3: Step-Path Scratch and Logits Residency

- [x] 3.1 Introduce persistent or pooled standard-step scratch sized from existing `step_scratch_len`, with zero per-token scratch allocation in GPU sessions.
- [x] 3.2 Define logits egress so GPU compute returns one host-visible 1D float32 logits buffer per token with at most one synchronization point.
- [x] 3.3 Keep `generate_embeddings_mojo` explicitly out of the generation-session pool unless an isolated embedding pool is introduced; do not regress current generation-session reuse.
- [x] 3.4 Checkpoint: review token-loop transfer/allocation budget and reject designs that require repeated Python array pointer extraction or cache copies.

### Phase 4: Validation and Operational Readiness

- [x] 4.1 Add deterministic parity tests focused on cache-dependent multi-token sequences, cache reset behavior, and session reuse.
- [x] 4.2 Add stress tests for long-context cache writes/reads, stable memory footprint, and repeated generation/reset cycles.
- [x] 4.3 Record performance evidence for allocation elimination, copy-count reduction, and step-latency improvement.
- [x] 4.4 Manual verification: run long-sequence generation on the standard path, confirm stable outputs and bounded memory, and capture copy/sync counters.
- [x] 4.5 Checkpoint: chapter signoff bundle includes memory-lifecycle notes, byte formulas, perf evidence, and the correctness-gate handoff contract.

## TDD-Oriented Task Checklist

- [x] RED: add failing tests in `src/py/tests/test_mojo_core.py` that expose host-resident KV assumptions, cache object/handle churn, and non-deterministic reset behavior.
- [x] RED: add failing standard-path layer tests in `src/mo/tests/test_layers.mojo` for cache read/write ordering and multi-token sequence reuse.
- [x] GREEN: implement device-resident KV lifecycle and session-owned scratch pools so existing and new tests pass.
- [x] GREEN: implement explicit transfer/sync policy with deterministic output parity and stable reset semantics.
- [x] REFACTOR: isolate cache/scratch ownership and handle plumbing so generation hot path has no repeated pointer extraction or duplicate allocation logic.
- [x] REGRESSION: run targeted `src/py/tests/test_mojo_core.py` plus standard cache-sensitive `src/mo/tests/test_layers.mojo`.
