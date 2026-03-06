# Flow: performance-baselines_20260222

## Specification

### Goal

Establish reproducible performance baselines and release guardrails for embedding and generation so first-release performance claims are evidence-backed.

### Code Analysis Summary

1. `src/py/mogemma/model.py` generation and embedding paths are measurable at wrapper level and core-coupled level.
2. There is currently no standardized benchmark harness or threshold policy in repo-level tooling.
3. Prior architecture patterns emphasize zero-copy and async bridging, but release evidence requires explicit measurable outcomes.
4. README currently references capabilities but does not publish formal baseline metrics.

### Requirements

1. Benchmark workloads must be reproducible and documented.
2. Metrics must include latency/throughput for embedding and generation paths.
3. Baselines must include environment details to avoid misleading comparisons.
4. Release thresholds must be defined for regression detection.
5. Performance report must be concise, factual, and non-promotional.

### Non-goals

1. Exhaustive micro-optimization work for every code path.
2. Hardware-vendor-specific tuning for this release.
3. Benchmarking every possible model variant.

## Implementation Plan

### Phase 1: Benchmark Scope and Methodology

- [x] 1.1 Define benchmark environments (hardware, OS, Python, Mojo versions).
- [x] 1.2 Define representative embedding workloads (single input, batch input, tokenized input).
- [x] 1.3 Define representative generation workloads (short prompt, long prompt, streaming).
- [x] 1.4 Define measurement methodology (warmup, run count, aggregation statistics).

### Phase 2: Benchmark Harness Implementation

- [x] 2.1 Implement benchmark scripts/tools under project tooling conventions.
- [x] 2.2 Ensure scripts output structured metrics suitable for comparison over time.
- [x] 2.3 Add deterministic configuration options to reduce benchmark noise.
- [x] 2.4 Validate benchmark scripts run in clean environments.

### Phase 3: Baseline Capture and Analysis

- [x] 3.1 Capture embedding baseline metrics across defined workloads.
- [x] 3.2 Capture generation baseline metrics across defined workloads.
- [x] 3.3 Record variance and identify outlier behavior.
- [x] 3.4 Document baseline limitations and caveats.

### Phase 4: Thresholds and Regression Guardrails

- [x] 4.1 Define release thresholds for acceptable regression per metric.
- [x] 4.2 Define fail conditions and triage path when thresholds are exceeded.
- [x] 4.3 Add lightweight regression check in CI or release check pipeline.
- [x] 4.4 Document when performance checks are mandatory vs advisory.

### Phase 5: Reporting and Release Integration

- [x] 5.1 Produce concise baseline report with commands and environment metadata.
- [x] 5.2 Ensure report language avoids superlatives and unsupported interpretation.
- [x] 5.3 Link baseline report in release evidence checklist.
- [x] 5.4 Capture Beads evidence for reproducibility.

## Test Plan

1. Benchmark harness dry-run commands for embedding and generation workloads
2. `make test`
3. `make lint`

## Validation Criteria

1. Baseline metrics are reproducible with documented commands and environment.
2. Thresholds and regression policy are explicit.
3. Release evidence includes performance report and guardrail policy.
4. Performance wording in release docs is factual and bounded.

## Risks and Mitigations

1. Risk: Benchmark noise masks real regressions.
   Mitigation: Standardize environment, warmup, and repeated measurement policy.
2. Risk: Thresholds are either too strict or too loose.
   Mitigation: Use baseline variance to set initial thresholds and document review criteria.
