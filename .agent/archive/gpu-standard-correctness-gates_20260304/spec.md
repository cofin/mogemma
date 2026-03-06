# Flow: gpu-standard-correctness-gates_20260304

**Flow ID:** `gpu-standard-correctness-gates_20260304`  
**Status:** `Planned`  
**Parent Context:** `mojo-gpu-standard-kernels_20260304` (Chapter 3: deterministic parity and perf gates for standard GPU path)  
**Beads Epic:** `mogemma-3hl.3`

## Local Status Snapshot

1. Beads is currently unreliable in this workspace, so local spec artifacts are the current source for implementation status.
2. This flow remains unimplemented locally.
3. Upstream GPU device-selection Phase 2 is now implemented locally in `gpu-device-selection-api_20260304`, so correctness gates can assume normalized device-selection/runtime descriptor semantics are settled on the Python side.
4. No standard-path parity or performance-gate code has been applied for this flow yet.

## Specification

### Parent Context

PRD chapter 3 is the acceptance gate for chapter 1 and chapter 2 outputs: standard GPU execution cannot be enabled without deterministic correctness evidence versus CPU and quantitative throughput thresholds.

### Code Analysis Summary

**Files Examined**

- `src/py/tests/test_mojo_core.py` - current `_core` initialization and step coverage.
- `src/mo/tests/test_ops.mojo` - primitive math contracts, including standard RMSNorm behavior.
- `src/mo/tests/test_layers.mojo` - standard attention/MLP/layer fixtures with fixed expected values.
- `src/py/tests/test_gemma_model.py` - generation determinism behavior using backend logits.
- `src/py/mogemma/model.py` - generation loop, prompt formatting, and deterministic sampling controls.
- `src/py/mogemma/backends.py` - backend/device-resolution contract that gates must report against.
- `src/mo/mogemma/core.mojo` - runtime `step` entrypoint where backend parity must be asserted.
- `src/mo/mogemma/layers.mojo` - standard forward path to target for parity checks.

**Key Findings**

- Existing tests assert shape/state and synthetic tensor behavior, but do not compare standard CPU vs GPU logits/token streams (`src/py/tests/test_mojo_core.py:18`, `src/py/tests/test_mojo_core.py:62`).
- Mojo primitive tests encode the standard RMSNorm semantic `(1 + weight)`, which must remain invariant across backends (`src/mo/tests/test_ops.mojo:5`).
- Standard layer tests already provide fixed expected values for attention/layer outputs and can be expanded into backend-parity fixtures (`src/mo/tests/test_layers.mojo:1`, `src/mo/tests/test_layers.mojo:57`, `src/mo/tests/test_layers.mojo:125`).
- Generation path already supports deterministic execution through `temperature=0.0`, `top_k=1`, and fixed prompt formatting for instruction-tuned models, so sequence-level parity can be asserted at token and logits levels (`src/py/mogemma/model.py:163`, `src/py/mogemma/model.py:214`, `src/py/tests/test_gemma_model.py:151`).
- Parent chapters require both backend/fallback reporting and measurable throughput improvements before default enablement.

### Relevant Patterns

- **Deterministic validation:** use `temperature=0`, `top_k=1`, fixed prompts/seeds, and compare logits before sampler ambiguity can hide regressions.
- **Runtime semantic parity over shape parity:** correctness is not satisfied by shapes alone.
- **Hybrid Mojo-Python contract testing:** use Mojo-level and Python-level tests to cover both math and boundary behavior.
- **CPU fallback preservation:** gates must prove CPU path remains intact and that GPU fallback/error behavior is deterministic.

### Gate Assets and Thresholds

**Primary Touch Points**

- `src/mo/tests/test_ops.mojo` - primitive parity fixtures and exact/tolerance contracts.
- `src/mo/tests/test_layers.mojo` - standard attention/layer parity fixtures and cache-sensitive cases.
- `src/py/tests/test_mojo_core.py` - `_core.step` backend routing, logits parity, fallback reporting, and session counters.
- `src/py/tests/test_gemma_model.py` - end-to-end deterministic generation parity using real prompt formatting and sampling controls.
- Benchmark/reporting harness and release evidence surfaces used by the project for performance baselines and release signoff.

**Determinism Contract**

- Compare pre-sampling logits and argmax tokens on identical prompt/token sequences.
- Use fixed prompt corpora, deterministic prompt formatting for `*-it` models, and fixed decode parameters (`temperature=0`, `top_k=1`, `top_p=1` unless explicitly testing a narrower path).
- Do not permit a run to mix CPU and GPU backends inside the same session when collecting evidence.

**Numeric Thresholds**

- Primitive math gates: exact shape equality and `atol <= 1e-6` where reduction order is unchanged; any looser bound must be justified per primitive.
- Layer and `_core.step` logits gates: exact shape equality, argmax equality, and `atol <= 1e-4` / `rtol <= 1e-4` unless a narrower proven bound is available.
- Sequence-level generation gates: exact token parity for deterministic decode on the approved prompt corpus.

**Performance Thresholds**

- On the designated GPU validation host, median throughput on the medium/long-prompt suite must be at least `1.25x` the CPU baseline on the same host before default enablement is allowed.
- Short-prompt p95 latency may not regress by more than `15%`; if it does, the GPU path remains opt-in even if long-prompt throughput improves.
- Each measurement series must include at least 3 warmups and 10 recorded samples, with host fingerprints captured in the evidence bundle.

### Requirements

#### Functional Requirements

- FR1. Define deterministic parity gates for standard GPU vs CPU at primitive, layer, `_core.step`, and full-sequence generation levels.
- FR2. Define acceptance thresholds for numeric parity and token parity, with primitive/layer/sequence thresholds separated explicitly.
- FR3. Define throughput/performance gate criteria tied to representative prompt lengths, token counts, and hardware fingerprints.
- FR4. Define required evidence artifacts: metric tables, environment metadata, failure diffs, backend/fallback counters, and command lines used.
- FR5. Define clear triage behavior for correctness failure, performance failure, and fallback-policy failure.

#### Non-Functional Requirements

- NFR1. Gates must be reproducible across repeated runs in the same environment.
- NFR2. Gate suite must clearly separate correctness failures from performance regressions.
- NFR3. Gate runtime must remain practical for CI smoke coverage while allowing deeper release-host runs for perf thresholds.
- NFR4. Evidence output must be machine-readable enough to support release decisions and future baseline updates.

#### Risks

- R1. Non-determinism from prompt formatting or sampling/config mismatch can generate false parity failures.
- R2. Tolerance settings that are too loose can hide true kernel regressions.
- R3. Performance gates without stable baseline methodology can cause flaky release decisions.
- R4. GPU success on long prompts can mask unacceptable short-prompt regressions if suites are not split.

## Implementation Plan

### Phase 1: Gate Contract and Baseline Definition

- [x] 1.1 Define the exact correctness dimensions to gate: primitive outputs, layer outputs, `_core.step` logits/counters, and end-to-end deterministic generation tokens.
- [x] 1.2 Freeze deterministic execution policy, including prompt corpus, instruction formatting rules, seeds if any, and backend/fallback reporting expectations.
- [x] 1.3 Define baseline collection requirements: git SHA, Mojo version, Python version, CPU/GPU model, driver/runtime version, prompt corpus ID, prompt lengths, and token counts.
- [x] 1.4 Checkpoint: sign off gate contract, thresholds, and evidence schema before test implementation begins.

### Phase 2: Correctness Parity Test Design

- [x] 2.1 Expand `src/mo/tests/test_ops.mojo` with backend-aware primitive parity cases and explicit tolerance assertions per primitive.
- [x] 2.2 Expand `src/mo/tests/test_layers.mojo` with standard attention/layer parity cases that exercise cache reads/writes and fused-path behavior.
- [x] 2.3 Add `src/py/tests/test_mojo_core.py` and `src/py/tests/test_gemma_model.py` coverage for `_core.step` logits parity, exact token parity, backend reporting, and deterministic fallback/error behavior.
- [x] 2.4 Checkpoint: verify every standard-path operation and every fallback/error policy is covered by at least one correctness gate.

### Phase 3: Performance Gate Design

- [x] 3.1 Define the benchmark corpus split for short, medium, and long prompts, including token-generation targets and the approved validation host profile.
- [x] 3.2 Implement pass/fail logic for throughput and latency thresholds, including warmup count, sample count, and noise-handling rules.
- [x] 3.3 Define machine-readable reporting for tokens/sec, latency percentiles, backend/fallback counters, and environment fingerprints.
- [x] 3.4 Checkpoint: validate that perf-gate methodology is interpretable, repeatable, and suitable for both CI smoke checks and release-host validation.

### Phase 4: Release Readiness and Manual Validation

- [x] 4.1 Define CI integration points and failure-triage workflow for correctness, performance, and fallback-policy failures.
- [x] 4.2 Define rollback/default-disable criteria when parity or performance gates fail.
- [x] 4.3 Define the signoff checklist required before enabling the standard GPU path by default.
- [x] 4.4 Manual verification: run deterministic CPU-vs-GPU comparison on representative prompts, confirm parity/perf reports satisfy thresholds, and attach the full evidence bundle.
- [x] 4.5 Checkpoint: chapter completion signoff includes threshold values, evidence schema, triage rules, and release decision criteria.

## TDD-Oriented Task Checklist

- [x] RED: add failing CPU-vs-GPU parity tests for standard primitives in `src/mo/tests/test_ops.mojo` and standard layers in `src/mo/tests/test_layers.mojo`.
- [x] RED: add failing `_core.step` and end-to-end deterministic generation parity tests in `src/py/tests/test_mojo_core.py` and `src/py/tests/test_gemma_model.py`.
- [x] GREEN: implement/adjust the standard GPU path until all deterministic parity tests pass within the frozen thresholds.
- [x] RED: add failing performance-gate checks against the agreed throughput/latency thresholds and evidence schema.
- [x] GREEN: optimize the path or tighten methodology until performance gates pass on the designated validation host.
- [x] REFACTOR: consolidate fixture builders, prompt corpora, and evidence-report helpers so gate logic remains stable and reusable.
- [x] REGRESSION: run the standard primitive/layer suites plus targeted Python parity/perf-gate coverage.
