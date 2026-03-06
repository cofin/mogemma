# Learnings: Cleanup & hatch-mojo Migration

## [2026-02-25] - Build system migration

- **Implemented:** Migrated from custom `tools/hatch_build.py` to declarative `hatch-mojo` plugin config.
- **Learnings:**
  - Declarative build config in `pyproject.toml` is cleaner than custom hook files.
  - Deleting custom hooks breaks tests with hardcoded file paths — rewrite to parse TOML.
  - All references to deleted files must be grep-validated.
  - Pattern documentation can become stale — review during cleanup.
