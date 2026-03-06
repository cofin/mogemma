# Flow: pypi-cibuildwheel_20260225

## Specification

### Code Analysis Summary
The PyPI upload action fails with the HTTP 400 error `Binary wheel 'mogemma-...linux_x86_64.whl' has an unsupported platform tag 'linux_x86_64'.` because Linux wheels uploaded to PyPI require `manylinux` or `musllinux` compatibility tags. The current workflow in `.github/workflows/publish.yml` and `.github/workflows/test-build.yml` runs a direct `uv build --wheel`, which produces a generic host-bound Linux tag. To solve this and expand support to Windows, we must adopt `cibuildwheel`.

- **Affected Files:**
  - `.github/workflows/publish.yml`
  - `.github/workflows/test-build.yml`
  - `pyproject.toml` (configuration entry)

### Requirements
- Replace manual `uv build` matrix steps with `cibuildwheel` in GitHub Actions.
- Ensure the matrix properly supports Linux (`manylinux`), macOS, and Windows.
- Update `pyproject.toml` to configure `cibuildwheel` (`[tool.cibuildwheel]`) to:
  - Run tests during the wheel build step.
  - Setup Mojo CLI within the `cibuildwheel` environment prior to running the `hatch-mojo` backend (likely via a `CIBW_BEFORE_BUILD` or `CIBW_BEFORE_ALL` command using `uv`).
  - Configure `build-frontend = "build"` or `"uv"`.

## Implementation Plan

### Phase 1: Configure `cibuildwheel`
- [x] 1.1 Update `pyproject.toml` to include a `[tool.cibuildwheel]` section. Configure `before-all` / `before-build` to ensure the Mojo SDK and `uv` can be used to build the extension.
- [x] 1.2 Restructure `.github/workflows/publish.yml` to replace `build-wheel-mojo` and `test-wheels` with the official `pypa/cibuildwheel@v2` action for Linux, macOS, and Windows runners.
- [x] 1.3 Apply the exact same changes to `.github/workflows/test-build.yml` so PR branches test the new matrix correctly.

### Phase 2: Testing and Repair
- [-] 2.1 Verify that `hatch-mojo` successfully invokes the `mojo` compiler inside the Linux (Docker) and Windows `cibuildwheel` environments.
- [-] 2.2 Verify that the generated wheels correctly contain `manylinux` tags.