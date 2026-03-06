# Knowledge: Cleanup & hatch-mojo Migration

> Flow: cleanup-and-migrate | Archived 2026-02-25

## Key Changes

Migrated from custom `tools/hatch_build.py` to declarative `hatch-mojo` plugin config in `pyproject.toml`. Updated Makefile, tests, and patterns.

## Follow-up Alignment (2026-03-02)

- Packaging baseline updated to `hatch-mojo>=0.1.5`.
- `bundle-libs` moved to canonical config (`true`) instead of Linux-only CI toggle.
- Stale macOS empty repair override removed.
- Linux-specific modular-lib staging and retag repair retained with explicit rationale pending further CI evidence.

## Patterns

- **Declarative build config:** `[[tool.hatch.build.targets.wheel.hooks.mojo.jobs]]` in `pyproject.toml` replaces custom hook files.
- **Test rewrite pattern:** Parse TOML -> verify config structure instead of hardcoded file I/O.
- **Documentation sync:** Pattern documentation can become stale — review during cleanup phases.
- **Canonical policy over CI mutation:** Prefer stable TOML policy over runtime `sed` edits when upstream support is available.

## Gotchas

- Deleting custom build hooks breaks tests that hardcoded file paths.
- Build system tests are fragile — they couple to implementation details.
- All references to deleted files must be grep-validated.
