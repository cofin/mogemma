# Knowledge: API max_tokens Rename

> Flow: api-max-tokens_20260225 | Archived 2026-02-25

## Key Changes

Renamed `max_new_tokens` -> `max_tokens` across entire codebase: config, model, tests, CLI tools, docs, baseline JSON files. Breaking change, no backwards compat.

## Patterns

- **API renaming scope:** Parameter renaming touches config, model, tests, tools, docs, baselines — all must update together.
- **Industry-standard naming** preferred over project-specific naming.
- **Grep-based validation** needed to catch all occurrences across codebase.

## Gotchas

- Breaking change with no backwards compatibility.
- Documentation baselines (JSON files) must be updated to avoid test failures.
- CLI argument parsing must match API changes.
