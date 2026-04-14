# CI and release

## Workflows (`.github/workflows/`)

| File | Trigger | Purpose |
|---|---|---|
| `ci.yml` | PR + push to main | Lint, type-check, test matrix |
| `pr-title.yml` | PR title change | Enforce conventional-commits format |
| `publish.yml` | GitHub release | Build + publish to PyPI via trusted publishing |
| `perf-benchmark.yml` | manual dispatch | Advisory benchmarks, 90-day artifact retention |

## CI structure

Parallel jobs (independent, no cross-dependency):

- `lint-ruff` — `ruff check` + `ruff format --check`.
- `type-check-mypy` — strict.
- `type-check-pyright` — aligned expectations with mypy (not strict superset).
- `test-linux` — matrix 3.10–3.13, builds `_core.so` and runs pytest.
- `build-wheels` — per-platform (Linux x86_64/aarch64, macOS x86_64/arm64 via QEMU).
- `build-source` — sdist.
- `check-release` — gated on `workflow_dispatch`; runs parity gates + benchmarks.

## Release flow

1. Bump version via `bump-my-version`.
2. Commit + tag.
3. Push tag → GitHub release created.
4. `publish.yml` fires → wheels built → PyPI upload via OIDC.

## Rust downloader FFI contract

mogemma can optionally use a Rust-backed downloader (external crate) for faster concurrent fetches.

- Request/response structs defined in `src/py/mogemma/backends.py`.
- Cache root passed as config. Artifact selectors (tokenizer, weights, config.json) explicit.
- Retry + error taxonomy: network error, auth error, integrity error, cancellation.
- Tokenizer artifact selection: separate concern from weight selection.
- Concurrency: backpressure-aware, cancellable futures.

## obstore integration

- `HTTPStore` for Hugging Face.
- `GCSStore` for `gs://gemma-data`.
- `LocalStore` for cache mirroring.
- TensorStore used as fallback for local/GCS Orbax reads (not HTTP).

## Gotchas

- `CI=true` env var forces watch-mode tools (pytest-watch, ruff --watch) into single-shot mode.
- `check-release` is NOT part of PR gating — it runs only on manual dispatch to avoid perf-gate flakes blocking merges.
- Non-interactive is the rule: no prompts, no stdin reads in CI scripts.
- `gh` CLI authenticated via workflow's `GITHUB_TOKEN` — never hardcode tokens.
