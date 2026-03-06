# Knowledge: Release Setup

> Flow: release-setup | Archived

## Key Changes

Configured `bump-my-version` for version management. Created `publish.yml` and `test-build.yml` workflows. Added PyPI metadata and trusted publishing (OIDC).

## Patterns

- **Platform-specific wheels:** Pure Python fallback + Mojo-compiled variants. Code handles `_core = None`.
- **`uv build --wheel`** triggers hatch hook automatically if Mojo available.
- **Trusted publisher (OIDC):** Requires GitHub environment `pypi` configuration.

## Gotchas

- Mojo only supports **Linux x86_64** and **macOS arm64** — no Windows, no aarch64 Linux.
- `cibuildwheel` won't work directly with Mojo (hatch hook calls `mojo build` instead).
- Version management via `bump-my-version` only — no manual edits.
