# Knowledge: PyPI cibuildwheel

> Flow: pypi-cibuildwheel_20260225 | Archived 2026-02-25

## Historical Constraint: GLIBCXX Mismatch

The `mojo` pip package (`mojo-0.26.1.0`) requires `GLIBCXX_3.4.30` (GCC 12/libstdc++). However, `cibuildwheel`'s `manylinux_2_34_x86_64` image (AlmaLinux 9) only provides `GLIBCXX_3.4.29` (GCC 11).

**Result:** `mojo` compiler invocation inside `manylinux_2_34` container fails with `/lib64/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`.

## Current Status (updated 2026-03-02)

This is no longer treated as a hard blocker in mogemma's active configuration.

Current strategy:

- Keep Linux `before-all` libstdc++ bootstrap in `pyproject.toml`.
- Keep explicit Linux `MODULAR_LIB_DIR` staging in `before-build`.
- Keep Linux manylinux retag `repair-wheel-command`.
- Use `hatch-mojo>=0.1.5` and `bundle-libs = true` as steady-state packaging baseline.
- Remove stale macOS empty repair override; use default cibuildwheel behavior.

## Why this changed

Follow-up PRD `hatch-mojo-alignment_20260301` re-audited upstream behavior in `hatch-mojo` `v0.1.5` and removed outdated issue-#7-era assumptions while retaining proven Linux constraints with explicit rationale.
