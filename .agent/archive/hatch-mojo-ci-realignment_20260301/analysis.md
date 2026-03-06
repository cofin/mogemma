# Chapter 2 Implementation Report: hatch-mojo-ci-realignment_20260301

Date: 2026-03-02
Flow epic: `mogemma-227.2`

## Summary of applied changes

Implemented decisions from Chapter 1 in config + contract tests:

1. Updated `hatch-mojo` minimum from `>=0.1.4` to `>=0.1.5` in `pyproject.toml`.
2. Moved to canonical bundling policy:
   - Removed Linux `sed` toggle from `before-build`.
   - Set `[tool.hatch.build.targets.wheel.hooks.mojo] bundle-libs = true`.
3. Removed stale macOS empty repair override by deleting `[tool.cibuildwheel.macos] repair command override`.
4. Retained Linux `MODULAR_LIB_DIR` staging + custom retag repair command with updated comments and explicit rationale.
5. Updated packaging config contract test to assert:
   - Linux retag policy still present.
   - `bundle-libs` is `true`.
   - No explicit macOS repair override exists.

## Workflow compatibility assessment

No workflow file edits were required. Existing `.github/workflows/ci.yml` and `.github/workflows/publish.yml` invoke `cibuildwheel` and inherit behavior from `pyproject.toml`, so Chapter 2 policy changes are applied through config only.

## Validation evidence captured in this flow

- Static configuration alignment evidence:
  - `pyproject.toml` now encodes the intended steady-state policies directly.
  - `src/py/tests/test_mojo_namespace_layout.py` assertions updated to lock policy.
- Runtime CI/test execution evidence:
  - Not executed in this flow (no explicit test run request in this session).

## Rollback plan for retained overrides

Retained Linux-specific constraints:

1. `MODULAR_LIB_DIR` staging in `before-build`.
2. Linux custom retag `repair-wheel-command`.

Rollback/removal trigger:

- If CI shows stable wheel builds and runtime linking without explicit modular-lib staging or custom retag, remove these in a follow-up and align Linux to default repair flow.

Immediate rollback option:

- Revert this flow's `pyproject.toml` changes if wheel build regressions appear after merge.

