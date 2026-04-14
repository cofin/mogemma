# Learnings: API max_tokens Rename

## [2026-02-25] - Breaking API rename

- **Implemented:** Renamed `max_new_tokens` -> `max_tokens` across config, model, tests, CLI tools, docs, baselines.
- **Learnings:**
  - API parameter renaming touches many surfaces — all must update together.
  - Industry-standard naming preferred over project-specific naming.
  - Grep-based validation needed to catch all occurrences.
  - Documentation baselines (JSON files) must be updated to avoid test failures.
