# Flow: CI Setup

## Specification

### Code Analysis Summary

**Current state:**
- Zero `.github/workflows/` files exist
- Tests run via `make test` → `uv run pytest src/py/tests`
- Linting via `make lint` → ruff check/format, mypy, pyright
- Mojo compiler installed via `uv sync --group build` (uses Modular's extra index configured in `[tool.uv]`)
- Build hook (`tools/hatch_build.py`) compiles `_core.so` only for wheel target
- Python >=3.10, test suite: 24 tests + 1 skipped

**Reference:** sqlspec's `ci.yml` provides the template pattern — separate jobs for validate, mypy, pyright, and test matrix across Python versions.

**Key adaptation for mogemma:**
- Must configure Modular's extra index URL for Mojo deps: `extra-index-url = ["https://modular.gateway.scarf.sh/simple/"]`
- `uv sync --all-extras --dev` installs everything including Mojo
- No slotscheck (mogemma doesn't use `__slots__`)
- No docs build job (no docs yet)
- No pre-commit (not configured)

### Requirements

1. Create `.github/workflows/ci.yml` with jobs for lint, type-check, and test
2. Test matrix: Python 3.10, 3.11, 3.12, 3.13
3. Lint job: ruff check + ruff format --check
4. Type-check jobs: mypy + pyright (can be separate or combined)
5. All jobs use `uv` for dependency management
6. PR title linting with conventional commits (separate workflow)

### Non-goals
- No docs workflow (no docs exist yet)
- No Mojo build/test in CI (Mojo tests require Mojo compiler runtime, Python tests are sufficient)
- No pre-commit setup

---

## Implementation Plan

### Phase 1: CI Workflow
- [x] 1.1 Create `.github/workflows/ci.yml` with lint job (ruff check + format --check) [bb11033]
- [x] 1.2 Add mypy job to ci.yml [bb11033]
- [x] 1.3 Add pyright job to ci.yml [bb11033]
- [x] 1.4 Add test-linux job with Python version matrix (3.10, 3.11, 3.12, 3.13) [bb11033]
- [x] 1.5 Create `.github/workflows/pr-title.yml` for conventional commit PR title validation [bb11033]

### Phase 2: Validate
- [x] 2.1 Push branch and verify workflows run on GitHub (or dry-run validate YAML locally)
- [x] 2.2 Ensure all jobs pass on current codebase

### Validation Criteria
- All workflow YAML is valid
- Lint, type-check, and test jobs match the patterns in sqlspec's ci.yml
- Test matrix covers Python 3.10-3.13
- `uv sync --all-extras --dev` correctly installs all deps including Mojo on runners
