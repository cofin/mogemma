# Knowledge: CI Setup

> Flow: ci-setup | Archived 2026-02-15

## Key Changes

Created `.github/workflows/ci.yml` (lint, type-check, test matrix Python 3.10-3.13) and `pr-title.yml` (conventional commits). Modeled after sqlspec CI patterns.

## Patterns

- **Modular CI jobs:** Lint, type-check (mypy + pyright), and tests as separate, independent jobs.
- **uv for CI:** `uv sync --all-extras --dev` for dependency management in CI.
- **No Mojo in CI:** Python tests sufficient; Mojo compiler runtime not needed in CI.

## Gotchas

- Mojo compiler requires custom extra index URL: `https://modular.gateway.scarf.sh/simple/`.
- `uv` must be configured with Modular index or Mojo dependency resolution fails.
- No pre-commit configuration exists; no slotscheck needed.
