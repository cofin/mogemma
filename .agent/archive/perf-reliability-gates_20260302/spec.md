# Flow: perf-reliability-gates_20260302

## Specification

### Goal

Define objective acceptance gates that prove the migrated download path is faster and reliable enough for hard cutover.

### Code Analysis Summary

Files analyzed:

- `src/py/mogemma/hub.py`
- `src/py/tests/test_hub.py`
- `.agent/specs/obstore-cffi-architecture_20260302/spec.md`
- `.agent/specs/hub-hard-cutover_20260302/spec.md`
- `.agent/specs/obstore-to-mojo-cffi_20260302/prd.md`

Key findings:

1. The current obstore path uses Python-managed artifact concurrency, so the migrated path needs explicit CPU-overhead and throughput gates to prove the Rust cutover delivered real performance value.
2. Warm-cache resolution is already a cheap local-path check; the new gates need to protect that fast path from accidental regression while also validating cold-cache download gains.
3. The current test surface does not enforce deterministic behavior for partial download cleanup, retry exhaustion, or benchmark artifact schema consistency.
4. Release readiness will require a shared evidence format so CI, local rehearsals, and release checklists compare the same metrics and failure classes.

### Requirements

1. Define a benchmark plan for throughput/latency across representative model sizes and artifact layouts.
2. Define a reliability matrix covering retries, interruption recovery, partial-file cleanup, integrity failures, and deterministic error surfacing.
3. Define measurable pass/fail thresholds required for release readiness.
4. Define a shared evidence format for CI and release runbook inclusion.

### Benchmark Contract

The benchmark plan for this flow must cover these scenarios explicitly:

1. Warm-cache resolve: cached directory already present and valid, with zero network/downloader work expected.
2. Cold-cache standard model: safetensors model families where download time is dominated by artifact transfer.
3. Cold-cache Orbax/nano model: download plus post-commit conversion orchestration, measured separately so conversion cost is not confused with transport cost.
4. Retry-path run: injected transient failure that succeeds within retry budget.
5. Resume-after-interruption run: partial staging state exists from an interrupted attempt and the next run must recover without manual cleanup.

### Performance Threshold Policy

Unless a later revision tightens them further, these are the release-blocking defaults:

1. Warm-cache resolve must remain under 150 ms at P95 on the reference CI/local benchmark host and must perform zero downloader/network work.
2. Cold-cache standard safetensors models must show at least a 10% improvement in end-to-end download throughput versus the obstore baseline at P50, and must not regress more than 5% at P95.
3. Cold-cache Orbax/nano models must be no worse than the obstore baseline for download-plus-commit time while the conversion phase is measured and reported separately.
4. Python-process CPU time during cold-cache download must be materially reduced versus the obstore baseline; the initial gate is a 25% reduction target, reported as evidence even when the release threshold is waived for early bring-up.
5. Peak RSS growth in the Python process must not scale with artifact bytes transferred; evidence should show metadata-only overhead on the Python side.

### Reliability Threshold Policy

1. Permanent failure classes (`NOT_FOUND`, `INTEGRITY`, path validation, disk-full/permissions) must fail deterministically on every run with zero committed partial cache tree.
2. Transient network faults must either succeed within the documented retry budget or fail with a deterministic exhausted-retries category; silent fallback is not acceptable.
3. Interrupted or cancelled downloads must leave staging state that is either safely reusable or safely discarded on the next run, with no manual cleanup required.
4. Release readiness requires 100% pass rate for permanent-failure assertions and at least 95% success for transient-fault recovery scenarios where the fault clears within the retry budget.
5. Any benchmark or reliability result lacking the required environment fingerprint or git SHA is invalid evidence.

### Evidence Schema Requirements

Every benchmark or reliability artifact must include:

1. Git SHA, benchmark date, and host fingerprint (CPU, memory, OS, Mojo version, Python version).
2. Model identifier, model family, artifact count, bytes transferred, cache mode (warm/cold/resume), and whether conversion ran.
3. Total wall-clock time, transfer-only time when available, Python CPU time, retry count, and final outcome category.
4. Threshold comparison result and the exact baseline artifact used for comparison.

### Harness Scope Guardrails

1. The benchmark/fault harness itself is a required deliverable for this flow; release evidence is not sufficient if the reproduction path is not checked into the repository.
2. The harness must distinguish "resume-after-interruption" from full resumable-download support; the former is required, the latter remains out of scope unless promoted into a dedicated flow.
3. Any threshold waiver must be recorded alongside the artifact and expire explicitly; there is no indefinite manual override for release gates.

## Implementation Plan

### Phase 1: Benchmark framework

- [ ] 1.1 `mogemma-h4n.4.1` Define the benchmark matrix across warm cache, cold cache, resume-after-interruption, model-size tiers, and representative artifact-count distributions.
- [ ] 1.2 `mogemma-h4n.4.1` Define metrics and collection method: latency percentiles, throughput, Python CPU time, memory overhead, retry counts, and conversion-separated timing.

Phase 1 is incomplete unless the resulting matrix makes it obvious which runs compare against the obstore baseline and which runs isolate download-only versus conversion costs.

### Phase 2: Reliability framework

- [ ] 2.1 `mogemma-h4n.4.2` Define the reliability/fault-injection matrix and expected deterministic behaviors for transient faults, permanent faults, interruption, cancellation, and local I/O failures.
- [ ] 2.2 `mogemma-h4n.4.2` Define harness requirements for transient/permanent network failures, integrity mismatches, and partial-file recovery without manual cleanup.

Phase 2 must explicitly identify which faults are retryable, which are terminal, and how success versus failure is measured for each class.

### Phase 3: Acceptance contract

- [ ] 3.1 `mogemma-h4n.4.3` Define CI enforcement, release-blocking thresholds, allowed waivers, and the baseline-comparison algorithm.
- [ ] 3.2 Checkpoint: define evidence artifacts, storage location, and runbook-ready summary format for cutover signoff.

## TDD-Oriented Task Checklist

- [ ] RED: add failing schema tests for benchmark/reliability artifacts and missing environment fingerprints.
- [ ] RED: add failing fault-injection tests that assert cleanup guarantees and retryability classification.
- [ ] GREEN: implement the benchmark/fault harness and CI gate logic until the red tests pass.
- [ ] REFACTOR: keep artifact generation and baseline-comparison code readable while preserving machine-readable outputs.
- [ ] REGRESSION: rerun benchmark and fault suites against both the obstore baseline and the migrated downloader path.

### Required Validation Evidence

1. A benchmark matrix document tied to concrete model IDs or tiers.
2. A reliability matrix document tied to concrete injected fault classes and expected outcomes.
3. A schema example for benchmark/reliability JSON artifacts.
4. A release checklist entry showing exactly which thresholds block the hard cutover.

## Task Acceptance Criteria

### `mogemma-h4n.4.1` Benchmark matrix and baselines

1. Defines benchmark scenarios for warm cache, cold cache, mixed artifact sets, and resume-after-interruption runs.
2. Defines model-size tiers and artifact profiles representative of production usage, including Orbax/nano families.
3. Produces a baseline metrics format for before/after migration comparison and distinguishes transfer-only from conversion-included timing.

### `mogemma-h4n.4.2` Reliability and fault-injection matrix

1. Enumerates transient and permanent failure classes with expected outcomes and retryability rules.
2. Defines partial-file and interrupted-download recovery expectations, including staging cleanup guarantees.
3. Specifies deterministic user-facing error output for each failure class.

### `mogemma-h4n.4.3` CI enforcement and thresholds

1. Defines measurable pass/fail thresholds for warm-cache latency, cold-cache throughput, Python CPU overhead, and reliability outcomes.
2. Defines reliability pass/fail requirements and blocking logic in CI.
3. Defines policy for handling flaky signals versus deterministic failures and names which waivers, if any, are temporary.

### Evidence artifacts checkpoint

1. Defines required benchmark artifacts, storage location, and retention expectations.
2. Defines required reliability report artifacts and traceability to test runs.
3. Defines final release signoff checklist entries tied to objective gate outcomes and named baselines.
