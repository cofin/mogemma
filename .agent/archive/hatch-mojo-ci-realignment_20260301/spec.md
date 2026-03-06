# Flow: hatch-mojo-ci-realignment_20260301

## Specification

### Goal

Implement the approved Chapter 1 matrix decisions to remove stale workaround behavior and preserve only justified, current constraints for Linux/macOS wheel builds.

### Requirements

1. Align `build-system` hatch-mojo version strategy with audited upstream baseline.
2. Align `bundle-libs` policy and related Linux/macOS wheel repair commands with Chapter 1 decisions.
3. Remove workaround comments and logic that no longer match upstream behavior.
4. Keep CI workflow behavior consistent with resulting `pyproject.toml` semantics.
5. Update test assertions that currently encode superseded workaround expectations.

### Candidate Files (from Chapter 1 contract)

- `pyproject.toml`
- `.github/workflows/ci.yml`
- `.github/workflows/publish.yml`
- `src/py/tests/test_mojo_namespace_layout.py`

## Implementation Plan

### Phase 1: Configuration realignment

- [x] 1.1 Apply approved `build-system` and `[tool.hatch.build.targets.wheel.hooks.mojo]` policy changes.
- [x] 1.2 Apply approved Linux `cibuildwheel` changes (`before-build`, environment, repair command policy).
- [x] 1.3 Apply approved macOS `cibuildwheel` repair policy changes.

### Phase 2: CI and tests consistency

- [x] 2.1 Ensure workflow env/matrix behavior is compatible with new `pyproject.toml` decisions.
- [x] 2.2 Update packaging/config contract tests to match intended steady-state behavior.

### Phase 3: Validation evidence capture

- [x] 3.1 Capture quality-gate evidence for CI/build behavior required by this flow.
- [x] 3.2 Record rollback plan for any override that must be retained pending upstream or platform constraints.
