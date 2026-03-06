# Learnings: Performance Baselines

## [2026-02-22] - Baseline establishment

- **Implemented:** Reproducible benchmarks for embedding and generation workloads.
- **Learnings:**
  - Document environment metadata, warmup policy, and aggregation statistics.
  - Use baseline variance to set regression thresholds.
  - Hardware-specific tuning creates false baselines on other platforms.
  - Benchmarks on CI can diverge from local — clarify mandatory vs advisory checks.
