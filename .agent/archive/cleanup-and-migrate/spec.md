# Flow: cleanup-and-migrate

## Specification

### Code Analysis Summary

**What's already done on `feat/cleanup`:**
- `pyproject.toml`: Build system switched from custom hook to `hatch-mojo` declarative config
- `Makefile`: Build target uses `uv build --wheel` instead of direct `mojo build`
- `tools/hatch_build.py`: Deleted (git staged)
- `.agent/code-styleguides/{mojo,python}.md`: Deleted (git staged)

**What's broken:**
1. `src/py/tests/test_mojo_namespace_layout.py:14-20` — `test_build_targets_point_to_namespaced_core()` reads the deleted `tools/hatch_build.py` and asserts on its contents. This test will fail.

**What's stale:**
2. `.agent/patterns.md:14` — References `tools/hatch_build.py` as the build mechanism. Should reference `hatch-mojo` plugin config in `pyproject.toml`.

### Requirements

1. All tests must pass after changes
2. No references to `tools/hatch_build.py` remain in the codebase
3. `.agent/patterns.md` accurately describes the current build system
4. The flow registry (`.agent/flows.md`) is updated

## Implementation Plan

### Phase 1: Fix broken test

- [x] 1.1 Rewrite `test_build_targets_point_to_namespaced_core()` in `src/py/tests/test_mojo_namespace_layout.py` to verify the hatch-mojo config in `pyproject.toml` instead of reading the deleted `tools/hatch_build.py`. The test should:
  - Parse `pyproject.toml` and find the `[[tool.hatch.build.targets.wheel.hooks.mojo.jobs]]` entry
  - Assert the job input points to `src/mo/mogemma/core.mojo`
  - Assert the module is `mogemma._core`
  - Keep the existing `test_mojo_sources_live_in_package_namespace()` test unchanged (it's fine)

### Phase 2: Update documentation

- [x] 2.1 Update `.agent/patterns.md` — replace the `tools/hatch_build.py` reference in "Architecture Patterns" with `hatch-mojo` plugin config in `pyproject.toml`
- [x] 2.2 Update `.agent/flows.md` — append this PRD and flow to the registry

### Phase 3: Verify

- [x] 3.1 Run `uv run pytest src/py/tests/test_mojo_namespace_layout.py` to confirm the fixed test passes
- [x] 3.2 Grep the codebase for any remaining references to `hatch_build.py` or `tools/hatch_build`
