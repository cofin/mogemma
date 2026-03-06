# Flow: Release Setup

## Specification

### Code Analysis Summary

**Current state:**
- No version management tool configured
- Version hardcoded in `pyproject.toml` as `version = "0.1.0"`
- No `[tool.bumpversion]` config
- No publish workflow
- No `build` dependency group for release tools (only `mojo` currently)
- Custom hatch build hook compiles Mojo `_core.so` for wheel builds
- Hatch build hook uses `shutil.which("mojo")` or falls back to `.venv/bin/mojo`
- Modular extra index configured in `[tool.uv]`

**Reference:** sqlspec's `publish.yml` provides the template:
- `build-source` job (sdist)
- `build-wheels-standard` job (pure Python wheel)
- `build-wheels-mypyc` job (compiled wheels via cibuildwheel)
- `test-wheels` job (install + smoke test)
- `publish-release` job (PyPI trusted publisher)

**Key adaptation for mogemma:**
- Instead of mypyc, mogemma compiles Mojo → `_core.so` via the hatch build hook
- The standard `uv build --wheel` triggers the hook if mojo is available
- Platform-specific wheels need the Mojo compiler on each target OS/arch
- Mojo currently only supports Linux x86_64 and macOS arm64 — no Windows, no aarch64 Linux
- Pure Python fallback wheel should also be published (the code handles `_core = None`)
- `cibuildwheel` may not work directly since Mojo isn't a standard C compiler — the hatch hook calls `mojo build`

### Requirements

1. Configure `bump-my-version` in `pyproject.toml` (matching sqlspec pattern)
2. Add `bump-my-version` to the `build` dependency group
3. Add `release` and `pre-release` Makefile targets
4. Create `.github/workflows/publish.yml`:
   - Trigger: GitHub release published + workflow_dispatch
   - Build source distribution (sdist)
   - Build pure Python wheel (no Mojo, for fallback)
   - Build Mojo-compiled wheel(s) for supported platforms
   - Test wheel installation
   - Publish to PyPI via trusted publisher (OIDC)
5. Create `.github/workflows/test-build.yml` for testing the build config on PRs
6. Add PyPI project metadata to `pyproject.toml` (license, authors, classifiers, urls)
7. Configure GitHub environment `pypi` for trusted publishing

### Non-goals
- No Windows wheel (Mojo doesn't support Windows yet)
- No multi-arch matrix via cibuildwheel (Mojo availability is limited)
- No PGO optimization (Mojo compilation, not C)

---

## Implementation Plan

### Phase 1: Version Management & Metadata
- [x] 1.1 Add `bump-my-version` to `build` dependency group [98acd5f]
- [x] 1.2 Configure `[tool.bumpversion]` in `pyproject.toml` (matching sqlspec pattern) [98acd5f]
- [x] 1.3 Add PyPI project metadata: license, authors, classifiers, urls, readme [98acd5f]
- [x] 1.4 Add `release` and `pre-release` Makefile targets [98acd5f]

### Phase 2: Build Workflow
- [x] 2.1 Create `.github/workflows/publish.yml` with sdist + pure wheel + Mojo wheel jobs [7ac3f77]
- [x] 2.2 Add test-wheels job to verify installation [7ac3f77]
- [x] 2.3 Add publish-release job with PyPI trusted publisher (OIDC) [7ac3f77]
- [x] 2.4 Create `.github/workflows/test-build.yml` for PR validation of build config [7ac3f77]

### Phase 3: Validate
- [x] 3.1 Run `uv build --sdist` and `uv build --wheel` locally to verify [98acd5f]
- [x] 3.2 Test that `bump-my-version` works: `uv run bump-my-version show current_version` [98acd5f]

### Validation Criteria
- `uv build --sdist` produces a valid sdist
- `uv build --wheel` produces a wheel with `_core.so` when Mojo is available
- `uv run bump-my-version show current_version` returns `0.1.0`
- publish.yml YAML is valid and follows sqlspec's pattern
- PyPI metadata appears correctly in `uv build` output
