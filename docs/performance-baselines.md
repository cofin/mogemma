# Performance baseline notes (bp0.7)

This project keeps a deterministic, script-driven baseline for generation and embedding paths.

## Environment reporting contract

Each baseline run must capture:

- `python` version
- `platform` string
- `processor` value (or `unknown`)
- `accelerator` value (e.g. `none`, `cuda`)
- model path
- benchmark mode and round counts
- elapsed seconds
- throughput metric

The JSON output from the benchmark script includes these fields in machine-readable form.

## Runner Normalization
Different CI runners have different baselines. Baselines should be strictly separated by `runner_class` and `backend` (e.g., standard-ci/cpu, standard-ci/gpu). Do not compare metrics across different runners or different backends directly without normalization factors.

## Commands

1. `uv run python tools/benchmark.py --mode generation --rounds 30 --max-new-tokens 64 --backend cpu`
2. `uv run python tools/benchmark.py --mode embedding --rounds 30 --backend cpu`
3. `uv run python tools/benchmark.py --mode generation --rounds 30 --max-new-tokens 64 --backend gpu --device gpu`
4. `make benchmark` (wrapper for both generation and embedding modes)

CI/release validation now executes generation benchmarks in `.github/workflows/ci.yml` through the `check-release` job.

## Baseline artifacts

Latest captured snapshots are stored in this directory:

- `docs/baseline-generation.json`
- `docs/baseline-embedding.json`

Capture both with:

```bash
uv run python tools/benchmark.py --mode generation --rounds 30 --max-new-tokens 64 > docs/baseline-generation.json
uv run python tools/benchmark.py --mode embedding --rounds 30 > docs/baseline-embedding.json
uv run python tools/benchmark.py --mode generation --rounds 30 --max-new-tokens 64 --backend gpu --device gpu > docs/baseline-generation-gpu.json
```

## Baseline output

Example output:

```json
{
  "schema_version": 2,
  "backend": "gpu",
  "device": "gpu",
  "runner_class": "standard-ci",
  "metrics": {
    "calls_per_second": 123.4,
    "elapsed_s": 0.2456,
    "input_texts": 2
  },
  "environment": {
    "platform": "macOS-15.3-x86_64-i386-64bit",
    "processor": "Intel(R) Core(TM) i7-1068NG7",
    "accelerator": "cuda",
    "python": "3.12.3"
  },
  "mode": "embedding",
  "model_path": "benchmark-model",
  "measured_rounds": 30
}
```

## Regression Interpretation and Acceptance Thresholds

- Establish project-specific baselines before each major runtime or compiler update.
- Treat generation regression as:
  - More than `12%` drop in throughput from last baseline for identical hardware and backend.
  - GPU specific: Check if GPU drops below 1.25x of CPU baseline.
- Treat embedding regression as:
  - More than `15%` drop in calls/sec from last baseline.

If variance exceeds threshold, run a second capture and compare both results before declaring regression. Ensure thermal throttling isn't causing false positives.

## Reporting

- Store raw JSON output and invocation command in release evidence.
  - `docs/baseline-generation.json`
  - `docs/baseline-embedding.json`
- Keep all claims factual and avoid superlatives.

## Runtime-State Refactor Evidence (2026-03-05)

Commands executed:

1. `uv run python tools/benchmark.py --mode generation --rounds 50 --max-new-tokens 64 > docs/baseline-generation.json`
2. `uv run python tools/benchmark.py --mode embedding --rounds 50 > docs/baseline-embedding.json`

Captured metrics:

- Generation:
  - `tokens_per_second`: `6500.09526611428`
  - `elapsed_s`: `0.015230546006932855`
  - `rounds`: `50`
- Embedding:
  - `calls_per_second`: `74006.10586899596`
  - `elapsed_s`: `0.0006756199290975928`
  - `rounds`: `50`

## Performance report template

Use this template when assembling release evidence:

- Date/hardware:
- Baseline files: generation + embedding snapshots in `docs/`.
- Generation regression gate: `12%` throughput drop.
- Embedding regression gate: `15%` throughput drop.
- Manual trigger condition: rerun baseline for changes in core Mojo or tokenizer/tokenization path.

## Batched Memory Layout Evidence (2026-03-07)

Commands executed:

1. `uv run python tools/benchmark.py --mode embedding --rounds 10 --batch-size 8`

Captured metrics for batched inference (mock stub backend):

- Embedding (Batch Size 8):
  - `calls_per_second`: `~29000`
  - `elapsed_s`: `~0.00034`
  - `rounds`: `10`
  - `input_texts`: `8`

*Note: True latency characteristics will become apparent on the real weights via `mat_mat_mul` parallelization.*
