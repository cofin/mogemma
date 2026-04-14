# Learnings: Release Setup

## [2026-02-15] - Release pipeline configuration

- **Implemented:** bump-my-version, publish.yml, test-build.yml, PyPI trusted publishing (OIDC).
- **Learnings:**
  - Mojo only supports Linux x86_64 and macOS arm64 — no Windows, no aarch64 Linux.
  - `cibuildwheel` won't work directly with Mojo (hatch hook calls `mojo build`).
  - Pure Python fallback wheel must be published (code handles `_core = None`).
  - `uv build --wheel` triggers hatch hook automatically if Mojo available.
