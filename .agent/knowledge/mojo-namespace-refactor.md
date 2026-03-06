# Knowledge: Mojo Namespace Refactor

> Flow: mojo-namespace-refactor | Archived 2026-02-23

## Key Changes

Moved `core.mojo`, `layers.mojo`, `model.mojo`, `ops.mojo` into `src/mo/mogemma/`. Updated imports in source/tests and build targets in `Makefile` and `tools/hatch_build.py`.

## Patterns

- **Namespace refactors:** Validate with a dedicated layout/tooling test that checks both source paths and build entrypoint strings.
- **Import stability:** Keeping `-I src/mo` unchanged allows tests under `src/mo/tests/` to import `mogemma.*` without relocating test files.
