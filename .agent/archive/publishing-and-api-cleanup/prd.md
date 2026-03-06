# Master PRD: PyPI Publishing & API Cleanup

## Context
The current PyPI publishing workflow fails for Linux wheels because it attempts to upload wheels with the generic `linux_x86_64` tag instead of a compliant `manylinux` tag. Additionally, the project lacks Windows wheel support, and the `GenerationConfig` API uses a confusing parameter name (`max_new_tokens`) that needs to be updated to the industry standard (`max_tokens`).

## North Star Goal
Ensure reliable cross-platform wheel distribution (Linux, macOS, Windows) to PyPI via GitHub Actions, and provide a cleaner, industry-standard text generation API for users.

## Roadmap

### Chapter 1: `pypi-cibuildwheel_20260225`
**Description:** Migrate the GitHub Actions build matrix to use `cibuildwheel`. This standardizes building `manylinux` wheels for Linux, ensures macOS wheels are built correctly, and adds Windows support to the build matrix. We will need to configure `cibuildwheel` to ensure the Mojo SDK is installed in the build environments (especially Linux Docker containers and Windows hosts) before compilation.

### Chapter 2: `api-max-tokens_20260225`
**Description:** Refactor the `GenerationConfig` API to rename `max_new_tokens` to `max_tokens` across the entire codebase. This will be an immediate breaking change without backwards compatibility to ensure a clean codebase. All associated references, test configurations, CLI tools, and documentation baselines must be updated to match.

## Global Constraints
- **Testing:** No feature merges without ensuring the changes pass all existing quality gates (`uv run pytest`, `uv run ruff`).
- **Compatibility:** No backwards compatibility for the API cleanup (Chapter 2).
- **Mojo SDK:** The Mojo SDK is a build dependency for `hatch-mojo`. `cibuildwheel` Linux environments must have `curl` or `wget` accessible to run `modular install mojo` (or equivalent `uv` script if applicable) in a `CIBW_BEFORE_BUILD` hook.