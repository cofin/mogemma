# Knowledge: Quality Gates (Zero-Warning)

> Flow: quality-gates-zero-warning_20260222 | Archived 2026-02-22

## Key Changes

Removed soft-fail patterns (`|| echo`) from `make lint`. Minimized suppression surface. Added `make check-release` target. Aligned CI with strict local `make lint`.

## Patterns

- **Hard-fail gates:** Soft-fail checks create false-green signals; hard failure enforces accountability.
- **Suppression audit:** Inventory all `noqa`/`ignore`, justify remaining, remove dead suppressions.
- **CI-local parity:** Local green must mean CI green — ensure identical strictness.

## Gotchas

- Strict gating can expose large backlogs — scope fixes to release-critical scope.
- `pyproject.toml` ignore lists can drift from actual suppressions — audit both.
- Type checker strictness varies by tool — align mypy + pyright expectations explicitly.
