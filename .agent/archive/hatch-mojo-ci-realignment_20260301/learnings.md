# Learnings: hatch-mojo-ci-realignment_20260301

## 2026-03-02

- Encode final packaging policy directly in `pyproject.toml`; avoid mutating config during CI runs when a stable upstream baseline exists.
- Keep Linux-specific workarounds narrowly scoped and explicitly documented with exit criteria, instead of broad platform comments tied to resolved upstream issues.
- Use config-contract tests to guard packaging policy (`bundle-libs`, repair override presence/absence) so future CI tweaks do not silently reintroduce drift.

