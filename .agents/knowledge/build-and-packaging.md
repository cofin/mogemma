# Build and packaging

## Build system

- **hatchling** + **hatch-mojo** plugin (`>=0.1.8`).
- Plugin compiles `src/mo/mogemma/core.mojo` → `src/py/mogemma/_core.so` during wheel build.
- Build isolation: uv resolves dependencies; Mojo compiler comes from the custom extra index URL declared in `[tool.uv.index]`.
- Per-architecture jobs: `core-x86_64` (flags `--target-cpu x86-64-v3`), `core-aarch64` (`--target-cpu native`).
- `bundle-libs = true` is the **canonical setting** for both local and CI — not optional. Mojo runtime libs are bundled into the wheel.

## Python support matrix

- Declared: 3.10 – 3.14.
- CI tested: 3.10 – 3.13.
- `tomllib` on 3.11+; fall back to `tomli` on 3.10 via conditional import.

## cibuildwheel

### Targets

- Linux x86_64 + aarch64 (aarch64 via QEMU).
- macOS x86_64 + arm64.

### Skip patterns (gotcha)

**Skip identifiers use underscore, NOT hyphen:**

```toml
skip = ["*_i686", "*-musllinux_*", "pp*", "*-win32"]
```

Wrong (`*-i686`) silently does nothing because cibuildwheel matches `cp310-manylinux_i686` — underscore between arch components.

### `before-all` runs INSIDE the Docker container on Linux

`CIBW_BEFORE_ALL_LINUX` executes in the manylinux container, not on the host runner. Any tool it installs is scoped to the container. macOS `before-all` not used.

## manylinux + libstdcxx quirk

**The problem:** `manylinux_2_34` (AlmaLinux 9) ships GCC 11 → max `GLIBCXX_3.4.29`. Mojo runtime needs `GLIBCXX_3.4.30`+ (GCC 12+).

**What does NOT work:**
- `libstdcxx-ng` on conda-forge is now an **empty transitional package**. Use `libstdcxx` (no `-ng` suffix) for the real `libstdc++.so.6`.
- `gcc-toolset-13` on AlmaLinux 9 provides only static libs and headers — no runtime `libstdc++.so.6`.

**The workaround:**

```bash
# CIBW_BEFORE_ALL_LINUX (inside manylinux container)
download conda-forge libstdcxx-14.3.0 package
unpack → copy lib/libstdc++.so.6 → /lib64/
ldconfig
```

## Wheel repair

`auditwheel` refuses wheels that import `GLIBCXX` symbols newer than the manylinux baseline. Since hatch-mojo already bundles Mojo libs + patches RPATH, auditwheel's repair is unnecessary (and harmful).

**Override:** use `wheel tags --platform-tag manylinux_2_34_$(uname -m)` to retag only, skipping repair.

## PyPI publish

- `publish.yml` workflow triggers on GitHub release.
- Uses trusted publishing (OIDC) — no API tokens in secrets.
- `bump-my-version` manages version strings across pyproject + `__version__`.

## Local build

```bash
uv sync                 # install dev deps incl. Mojo compiler
make build              # wheel build
make install-dev        # editable install with compiled _core.so
```

## Hatch config reference

Key settings in `pyproject.toml`:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/py/mogemma"]

[tool.hatch-mojo]
bundle-libs = true
# per-job overrides for target-cpu in CI matrix
```

## Gotchas

- **Dependency optional → base promotion** breaks reproducibility: refresh `uv.lock` and validate in clean environment.
- **Dead code audit** required after policy/dependency changes — stale imports masquerade as passing.
- **Mojo compiler extra index** must be configured in `[tool.uv.index]` or CI will fail with unresolvable `mojo-compiler` dep.
- **Benchmark hardware bias:** don't hard-code perf thresholds tuned on one machine; treat them as advisory.
