# Knowledge: Performance Baselines

> Flow: performance-baselines_20260222 | Archived 2026-02-22

## Patterns

- **Reproducible benchmarks:** Document environment metadata, warmup policy, aggregation statistics.
- **Representative workloads:** Embeddings (single, batch, tokenized) and generation (short, long, streaming).
- **Regression thresholds:** Use baseline variance to set initial thresholds; define fail conditions and triage paths.
- **Factual reporting:** Bound performance claims to measured baselines only.

## Gotchas

- Benchmark noise masks real regressions — standardize environment, warmup, repeated measurement.
- Hardware-specific tuning creates false baselines on other platforms.
- Benchmarks on CI can diverge from local — clarify when perf checks are mandatory vs advisory.

## CI Automation

- `.github/workflows/perf-benchmark.yml` runs `make benchmark` on every push to `main` and on PRs.
- Uses synthetic stubs (no real model weights) — measures framework overhead, not model quality.
- Results uploaded as artifacts with 90-day retention. Purely informational — does not gate PRs.
- The `check-release` job in `ci.yml` also runs benchmarks but only on `workflow_dispatch`.
