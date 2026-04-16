# Quality gates

## Canonical commands

```bash
make lint           # ruff check + ruff format --check + mojo format
make test           # pytest (src/py/tests + src/mo/tests)
make type-check     # mypy + pyright (both must pass)
make check-all      # lint + test + coverage — canonical pre-PR gate
make check-release  # release preflight: check-all + benchmarks
make coverage       # pytest with coverage; outputs htmlcov/index.html
make smoke-test     # quick Mojo bridge smoke test
make build          # hatch-mojo compile core.mojo → _core.so (required after Mojo edits)
```

**Zero-warning policy:** hard-fail. No soft-fail `|| echo` patterns. If a
warning is truly justified, suppress it narrowly with a reason comment.

**Local-CI parity:** `make lint` must produce the same exit code as CI's
lint-ruff job. Do not let them drift.

## Ruff configuration (from `pyproject.toml`)

- `line-length = 120`
- `target-version = "py310"`
- `select = ["ALL"]`, `ignore = ["COM812"]` (handled by formatter)
- `fixable = ["ALL"]`, `extend-safe-fixes = ["TC"]`
- Format: `preview = true`, `skip-magic-trailing-comma = true`, `docstring-code-format = true`
- Pydocstyle: `convention = "google"`
- Source roots: `src/py/mogemma`, `src/py/tests`, `src/mo/tests`
- First-party: `mogemma`, `tests`

### Per-file overrides

- Tests (`src/py/tests/test_*.py`) relax `ANN202`, `ARG001`, `S101`, `PLR2004`, etc.

## Mypy

- `python_version = "3.10"`
- `packages = ["mogemma", "tests"]`
- `exclude = ["tmp/", ".tmp/"]`
- Strict mode on public API.

## Pyright

- `pythonVersion = "3.10"`, `venv = ".venv"`
- `include = ["src/py/mogemma", "src/py/tests", "src/mo/tests"]`
- Expectations aligned with mypy — not a strict superset.

## Mojo format

- Mojo's built-in formatter; line length 120.
- `mogemma-fmt-mojo.py` helper script exists in some setups.

## Suppression audit

When adding `# noqa`, `# type: ignore`, or similar:

1. State the reason in the comment (`# noqa: SLF001  # internal stub pattern — see orbax_loader.py`).
2. Narrow the suppression to the specific rule code, never blanket.
3. Periodically audit: dead suppressions (rule no longer flags) must be removed.

## Test matrix

- pytest: `src/py/tests/*` + `src/mo/tests/*` (mojo test harness).
- `PYTHONPATH` includes `src/py`.
- `anyio` for async tests (not asyncio-specific).

## Known flaky tests

- **`test_async_generate_stream`** — `assert len(tokens) > 0` fails intermittently. Pre-existing, not a regression. Do NOT block merges on it; fix is tracked separately.

## Coverage target

- Aim: >80% for all modules.
- `make coverage` produces `htmlcov/index.html`. Drop below target = investigation, not auto-fail.

## Tomllib compatibility

```python
import sys
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib
```

## Anti-patterns to reject on review

- `# type: ignore` without a rule code
- `except Exception` with no re-raise or log
- `Any` in public API signatures
- Silent fallback behavior (always prefer explicit branches + explicit errors)
- Placeholder stubs that "pass" tests via default return — use real fixtures
