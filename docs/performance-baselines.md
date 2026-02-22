# Performance baseline notes (bp0.7)

This project keeps a deterministic, script-driven baseline for generation and embedding paths.

## Environment reporting contract

Each baseline run must capture:

- `python` version
- `platform` string
- `processor` value (or `unknown`)
- model path
- benchmark mode and round counts
- elapsed seconds
- throughput metric

The JSON output from the benchmark script includes these fields in machine-readable form.

## Commands

1. `uv run python tools/benchmark.py --mode generation --rounds 30 --max-new-tokens 64`
2. `uv run python tools/benchmark.py --mode embedding --rounds 30`
3. `make benchmark` (wrapper for both generation and embedding modes)

CI/release validation now executes generation benchmarks in `.github/workflows/ci.yml` through the `check-release` job.

## Baseline artifacts

Latest captured snapshots are stored in this directory:

- `docs/baseline-generation.json`
- `docs/baseline-embedding.json`

Capture both with:

```bash
uv run python tools/benchmark.py --mode generation --rounds 30 --max-new-tokens 64 > docs/baseline-generation.json
uv run python tools/benchmark.py --mode embedding --rounds 30 > docs/baseline-embedding.json
```

## Baseline output

Example output:

```json
{
  "calls_per_second": 123.4,
  "elapsed_s": 0.2456,
  "environment": {
    "platform": "macOS-15.3-x86_64-i386-64bit",
    "processor": "Intel(R) Core(TM) i7-1068NG7",
    "python": "3.12.3"
  },
  "input_texts": 2,
  "mode": "embedding",
  "model_path": "benchmark-model",
  "rounds": 30
}
```

## Acceptance thresholds

- Establish project-specific baselines before each major runtime or compiler update.
- Treat generation regression as:
  - More than `12%` drop in throughput from last baseline for identical hardware.
- Treat embedding regression as:
  - More than `15%` drop in calls/sec from last baseline.

If variance exceeds threshold, run a second capture and compare both results before declaring regression.

## Reporting

- Store raw JSON output and invocation command in release evidence.
  - `docs/baseline-generation.json`
  - `docs/baseline-embedding.json`
- Keep all claims factual and avoid superlatives.

## Performance report template

Use this template when assembling release evidence:

- Date/hardware:
- Baseline files: generation + embedding snapshots in `docs/`.
- Generation regression gate: `12%` throughput drop.
- Embedding regression gate: `15%` throughput drop.
- Manual trigger condition: rerun baseline for changes in core Mojo or tokenizer/tokenization path.
