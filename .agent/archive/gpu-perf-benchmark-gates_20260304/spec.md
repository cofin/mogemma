# Flow: gpu-perf-benchmark-gates_20260304

**Flow ID:** `gpu-perf-benchmark-gates_20260304`  
**Status:** `planned`  
**Parent PRD:** `mojo-gpu-packaging-gates_20260304`  
**Parent Context:** PRD4 Chapter 2 (GPU benchmark and regression gates), building on CPU baseline harness and future GPU runtime path.

## Local Status Snapshot

1. Beads is currently unreliable in this workspace, so local spec artifacts are the current source for implementation status.
2. This flow remains unimplemented locally.
3. Upstream GPU device-selection Phase 2 is now implemented locally in `gpu-device-selection-api_20260304`, so benchmark schema/gate work can assume normalized device labels and runtime descriptor fields are available.
4. No benchmark harness, schema, CI-gate, or baseline-policy changes have been applied for this flow yet.

## Specification

### Code Analysis Summary

#### Files Examined
- `tools/benchmark.py` (benchmark harness behavior and schema)
- `Makefile` (benchmark/check-release command wiring)
- `.github/workflows/ci.yml` (current benchmark gate invocation)
- `docs/performance-baselines.md` (threshold and reporting policy)
- `docs/release-runbook.md` (release evidence workflow)
- `src/py/mogemma/config.py` (public `device` config contract)
- `src/py/mogemma/model.py` (generation/embedding runtime behavior)
- `src/mo/mogemma/core.mojo` (step/init behavior affecting measured throughput)
- `src/mo/mogemma/ops.mojo` and `src/mo/mogemma/layers.mojo` (kernel characteristics)
- `src/py/tests/test_mojo_core.py` (core shape/position contracts)

#### Key Findings
- Benchmark CLI defines `--max-new-tokens`, but benchmark execution reads `args.max_tokens`, which is inconsistent and will break contract correctness unless fixed (`tools/benchmark.py:135`, `tools/benchmark.py:144`, `tools/benchmark.py:154`).
- Benchmark harness intentionally stubs `_core`, tokenizer, and loader, so current throughput numbers are deterministic but not representative of real backend performance (`tools/benchmark.py:49-87`, `tools/benchmark.py:138-150`).
- Environment payload currently captures Python/platform/processor but no backend/device/GPU metadata, which is insufficient for GPU regression triage (`tools/benchmark.py:89-95`).
- CI release gate runs only generation benchmark mode and does not compare against stored baselines or embedding mode thresholds (`.github/workflows/ci.yml:167-168`).
- Baseline docs define CPU-style regression thresholds (12% generation, 15% embedding) but no backend-aware policy for GPU variance or runner class segmentation (`docs/performance-baselines.md:61-67`, `docs/performance-baselines.md:84-85`).
- Runtime hot path still reconstructs model wrappers inside each token step, which can skew performance gates if not stabilized before GPU threshold enforcement (`src/mo/mogemma/core.mojo:411-437`).
- Public configs already expose `device`, but runtime path currently does not enforce it, so benchmark plan must include explicit backend/device labeling independent of current runtime behavior (`src/py/mogemma/config.py:20-21`, `src/py/mogemma/config.py:54-55`).

### Relevant Patterns
- Deterministic validation pattern: use deterministic settings (`temperature=0`, `top_k=1`) for regression checks so gate failures are attributable to performance/code changes rather than sampling noise (`.agent/patterns.md`).
- Contract testing pattern: benchmark outputs should be schema-tested and gate-tested as machine-readable contracts.
- Runtime gotcha: model rebuild overhead in hot paths must be accounted for when setting baseline thresholds to avoid false regressions (`.agent/patterns.md`, `src/mo/mogemma/core.mojo:411-437`).

### Requirements

#### Functional Requirements
- Define a backend-aware benchmark contract supporting at minimum `cpu_mojo` and planned `gpu_mojo` modes with explicit device labeling.
- Define benchmark JSON schema additions required for GPU gates (backend, device id, runner class, accelerator metadata, warmup parameters).
- Define threshold policy per mode/backend/model tier, including how to compare against baseline snapshots.
- Define CI gating behavior: which jobs run where, pass/fail conditions, and artifact retention for failed gates.
- Define release evidence requirements that tie benchmark commands, artifacts, and thresholds into go/no-go decisions.
- Define the canonical CLI contract for token-count arguments and mode selection, including backward-compatible alias behavior if `--max-new-tokens` is retained temporarily while callers migrate to a single canonical flag.
- Define the machine-readable schema fields required in benchmark JSON v2, including `schema_version`, `backend`, `device`, `runner_class`, `warmup_rounds`, `measured_rounds`, `threshold_profile`, and accelerator environment metadata.
- Define the baseline keying scheme used for comparisons, at minimum `(mode, backend, runner_class, model_tier)`, so CPU, synthetic, and real GPU baselines cannot be mixed accidentally.

#### Non-Functional Requirements
- Benchmark runs must be deterministic enough for CI gating with explicit warmup/repeat strategy and variance bounds.
- Benchmark execution must remain bounded in runtime cost for PR and release workflows.
- Gate outputs must be machine-readable and easy to audit after failures.
- CPU benchmark path must remain available when GPU runners are unavailable.
- Benchmark modes must declare whether they are synthetic/stubbed or real-backend and must never silently download models or dependencies during CI measurement.
- PR validation must stay within the current lightweight benchmark envelope; real GPU performance gates run only on dedicated GPU-capable runners or release validation paths.
- Threshold methodology must record variance controls explicitly so a failed gate can be diagnosed as code regression versus runner noise.

#### Required Deliverables
- A schema-v2 benchmark payload contract with example output for generation and embedding modes.
- A comparison/failure payload contract that CI can surface as annotations or archived JSON for triage.
- A baseline inventory plan showing where per-backend, per-runner-class benchmark snapshots live and how they are refreshed after approved runtime changes.

#### Risks
- Synthetic benchmark stubs can hide real runtime regressions if not clearly separated from real-backend benchmarks.
- Shared CI runners may introduce high variance that causes flaky gates.
- GPU runner availability/cost may block mandatory gates if fallback strategy is undefined.
- Runtime state-refactor dependencies may shift performance baseline significantly, requiring controlled baseline recalibration.

## Implementation Plan

### Phase 1: Benchmark Contract Hardening
- [x] 1.1 Define benchmark CLI/options contract, including the canonical token-count flag, any temporary alias/deprecation behavior, backend/device flags, and explicit synthetic-vs-real mode selection.
- [x] 1.2 Define JSON schema v2 for benchmark output, enumerating required top-level fields, environment fields, threshold metadata, and per-mode metrics for both generation and embedding.
- [x] 1.3 Define strict separation between synthetic (stub) and real-backend benchmark modes, including which code paths may install stubs, which modes require cached real assets, and how outputs label that distinction.
- [x] 1.4 Checkpoint: approve schema and CLI contract before implementation.

### Phase 2: Regression Gate Logic Design
- [x] 2.1 Define baseline loading/comparison algorithm keyed by `(mode, backend, runner_class, model_tier)` for both generation and embedding metrics.
- [x] 2.2 Define threshold policy (default and override paths), warmup/repeat counts, and variance handling procedure so flaky runners cannot silently poison the gate.
- [x] 2.3 Define failure payload format for CI annotations and release evidence, including metric deltas, threshold names, baseline identifiers, runner metadata, and next-action guidance.
- [x] 2.4 Checkpoint: validate gate logic against representative historical baseline files and synthetic fixtures before wiring CI.

### Phase 3: CI Integration Plan
- [x] 3.1 Design CI job graph for benchmark gates, explicitly separating always-on CPU/synthetic gates from GPU-required release jobs and naming the owning workflow files (`.github/workflows/ci.yml`, release workflow, `Makefile` wrappers).
- [x] 3.2 Define artifact upload and retention rules for raw benchmark JSON, comparison reports, and any failure summaries needed for triage or release sign-off.
- [x] 3.3 Define policy for unavailable GPU runners, including which PR paths may skip, which release paths must block, and how skipped-versus-blocked outcomes are encoded in artifacts.
- [x] 3.4 Checkpoint: confirm CI runtime budget, runner availability assumptions, and reliability targets are met before implementation begins.

### Phase 4: Release and Manual Verification Plan
- [x] 4.1 Define how benchmark evidence is linked into the release checklist and runbook steps, including the exact artifact names operators must attach.
- [x] 4.2 Define manual benchmark verification procedure on target GPU host(s) before release cut, plus the CPU-only control run required to detect gate drift.
- [x] 4.3 Manual verification task: execute benchmark suite in both CPU and GPU modes, compare against baselines, and archive raw reports, comparison reports, and runner metadata for release sign-off.
- [x] 4.4 Checkpoint: confirm final go/no-go criteria are measurable and reproducible.

## TDD-Oriented Task Checklist
- [ ] Add failing benchmark contract tests in a dedicated Python test module for CLI parsing, alias behavior, and schema-field presence (including the `--max-new-tokens`/canonical token flag decision).
- [ ] Add failing tests for benchmark schema serialization, required metadata fields, and synthetic-vs-real mode labeling.
- [ ] Add failing tests for baseline comparison logic, threshold gate outcomes, and failure-payload structure.
- [ ] Add failing CI workflow checks for required benchmark gate invocation, artifact uploads, and skip/block behavior when GPU runners are unavailable.
- [ ] Implement the minimal harness/gate changes needed to satisfy the failing tests without making real GPU runners a hard dependency for ordinary PRs.
- [ ] Re-run benchmark-focused tests plus existing core contract tests until green.
- [ ] Refactor benchmark code paths for clarity while preserving deterministic outputs.
- [ ] Add regression tests to ensure CPU-only workflows remain supported when GPU is not present and that release-mode GPU artifacts are still machine-readable when the runner is unavailable.
