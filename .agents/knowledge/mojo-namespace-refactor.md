# Learnings - mojo-namespace-refactor

## [2026-02-23 20:58] - Namespace migration and tooling alignment

- **Implemented:** Moved `core.mojo`, `layers.mojo`, `model.mojo`, and `ops.mojo` into `src/mo/mogemma/`; updated Mojo imports in source/tests; updated build targets in `Makefile` and `tools/hatch_build.py`.
- **Files changed:** `src/mo/mogemma/*.mojo`, `src/mo/tests/*.mojo`, `Makefile`, `tools/hatch_build.py`, `src/py/tests/test_mojo_namespace_layout.py`.
- **Learnings:**
  - Mojo namespace refactors are safer when validated by a dedicated layout/tooling test that checks both source paths and build entrypoint strings.
  - Keeping `-I src/mo` unchanged allows tests under `src/mo/tests/` to import `mogemma.*` without relocating test files.
