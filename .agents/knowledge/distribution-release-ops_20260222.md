# Learnings: Distribution & Release Ops

## [2026-02-22] - Release pipeline validation

- **Implemented:** Validated wheel/sdist matrix. Created release runbook. Isolated-env smoke tests.
- **Learnings:**
  - Test both wheel and sdist in isolated environments before publish.
  - Codify release steps explicitly; eliminate implicit assumptions.
  - Platform-specific build failures close to release are costly — validate early.
  - Wheel includes `_core.so` — verify it exists before publish.
