# Learnings: Quality Gates (Zero-Warning)

## [2026-02-22] - Hard-fail enforcement

- **Implemented:** Removed `|| echo` soft-fails. Minimized suppression surface. Added `make check-release`.
- **Learnings:**
  - Soft-fail checks create false-green signals; hard failure enforces accountability.
  - Suppression audit: inventory all `noqa`/`ignore`, justify remaining, remove dead.
  - CI-local parity: local green must mean CI green.
  - `pyproject.toml` ignore lists can drift from actual suppressions — audit both.
