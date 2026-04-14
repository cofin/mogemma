# Performance baselines

## Harness

- `make benchmark` — runs the full benchmark suite locally.
- 20 rounds per scenario, warmup policy standardized.
- Synthetic deterministic stubs (no real weights): measures **framework overhead**, not model quality.
- Results → JSON artifacts (CI uploads, 90-day retention on `perf-benchmark.yml`).

## Scenarios

- **Generation short** — `max_tokens=64`, single-request.
- **Generation long** — streaming, longer sequences.
- **Embeddings single** — one text input.
- **Embeddings batch** — rectangular batch.
- **Embeddings tokenized** — pre-tokenized input (skips tokenizer path).

## Reproducibility rules

1. Standardized env (Python version, deps pinned, no ambient processes).
2. Warmup rounds discarded.
3. Repeated measurement → median + MAD reported.
4. Thresholds derived from baseline variance, NOT from single-run numbers.

## Reporting

- **Informational**: does NOT gate PRs.
- Hardware-specific bias: a threshold tuned on one machine will falsely fail on another. Document this in any perf-related review.
- Regressions tracked via artifact diff, not automated hard-fail.

## Baseline philosophy

- "Perf checks advisory vs mandatory" must be explicit in the check name. Mandatory checks = correctness (parity, unit). Advisory = speed.
- Don't conflate: a correctness regression should hard-fail; a latency regression should post a comment, not block merge.

## Related

- See [ci-and-release.md](ci-and-release.md) for how `perf-benchmark.yml` is wired.
- See [gpu-infrastructure.md](gpu-infrastructure.md) for the kernel path under test.
