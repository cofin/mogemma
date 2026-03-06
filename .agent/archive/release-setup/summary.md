# Archive Summary: release-setup

**Archived:** 2026-02-22
**Status:** Complete
**Beads Epic:** mogemma-uqt.2

## Deliverables
- `[tool.bumpversion]` config in pyproject.toml (sqlspec pattern)
- PyPI metadata: MIT license, authors, classifiers, urls
- `bump-my-version` added to build dependency group
- `release` and `pre-release` Makefile targets
- `.github/workflows/publish.yml` — sdist + pure wheel + Mojo wheel + test + PyPI OIDC publish
- `.github/workflows/test-build.yml` — PR validation of build config
- `MOGEMMA_SKIP_MOJO=1` env var in hatch build hook

## Elevated Patterns
- Build Hook Skip: MOGEMMA_SKIP_MOJO=1 for pure Python wheel builds
- Dependency Groups: include-group composition pattern
