# Master PRD: CI, Release & README

## Context

mogemma is preparing for its v0.1.0 launch. The codebase is clean (launch-cleanup complete), but has zero CI/CD infrastructure, no version management, no PyPI publishing, and no README. This PRD sets up the full GitHub-based development and release pipeline modeled after sqlspec's proven patterns, adapted for the Mojo build toolchain.

**North Star:** A developer can `pip install mogemma`, the package is automatically tested/linted on every PR, and releases are published to PyPI via GitHub release tags with compiled Mojo wheels.

**Key Constraint:** The Mojo compiler is pip-installable from Modular's index (`https://modular.gateway.scarf.sh/simple/`). The `[tool.uv] extra-index-url` is already configured. The custom hatch build hook (`tools/hatch_build.py`) compiles `_core.so` during `uv build --wheel`.

## Roadmap

### Chapter 1: `ci-setup`
**CI Workflows** — Test, lint, and type-check on every PR and push to main.

### Chapter 2: `release-setup`
**Release Pipeline** — Version management with bump-my-version, wheel building (pure + Mojo-compiled), PyPI publishing via trusted publisher.

### Chapter 3: `project-readme`
**README** — Friendly project documentation with features, installation, and usage examples.

## Global Constraints

- **Owner:** `cofin/mogemma` on GitHub
- **License:** MIT
- **Python:** >=3.10
- **Patterns:** Follow sqlspec's CI/CD patterns adapted for Mojo
- **Build tool:** hatchling with custom Mojo build hook
- **Package manager:** uv
