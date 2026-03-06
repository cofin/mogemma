# Knowledge: hatch-mojo-ci-realignment_20260301

**Completed:** 2026-03-06
**Archived:** 2026-03-06

## Topics
CI, pyproject.toml, packaging policy, bundle-libs, config-contract tests, hatch-mojo

## Learnings
- Encode final packaging policy directly in `pyproject.toml`; avoid mutating config during CI runs when a stable upstream baseline exists.
- Keep Linux-specific workarounds narrowly scoped and explicitly documented with exit criteria, instead of broad platform comments tied to resolved upstream issues.
- Use config-contract tests to guard packaging policy (`bundle-libs`, repair override presence/absence) so future CI tweaks do not silently reintroduce drift.

## Summary
Realigned pyproject.toml and CI workflow build behavior with parity decisions and the current hatch-mojo baseline. Established config-contract tests to prevent silent packaging policy drift and narrowly scoped Linux-specific workarounds with explicit exit criteria.
