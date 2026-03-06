# Chapter 1 Implementation Report: hatch-mojo-upstream-parity_20260301

Date: 2026-03-02
Flow epic: `mogemma-227.1`

## 1) Active workaround inventory and provenance

### A. `build-system` hatch-mojo floor

- Current: `hatch-mojo>=0.1.4` in `pyproject.toml`.
- Provenance: introduced in `10e6e8c` after Linux wheel repair changes.
- Original rationale: pin to an upstream release that bundled runtime libs.

### B. Linux explicit modular lib staging + env override

- Current:
  - `[tool.cibuildwheel.linux].environment = { MODULAR_LIB_DIR = "/tmp/modular-lib" }`
  - `before-build` copies `modular/lib` into `/tmp/modular-lib`.
- Provenance: introduced in `10e6e8c` with comment tied to `cofin/hatch-mojo#7`.
- Original rationale: avoid runtime lib discovery failures in build env.

### C. Linux `sed` toggle for `bundle-libs`

- Current: `before-build` runs `sed -i 's/bundle-libs = false/bundle-libs = true/' pyproject.toml`.
- Provenance: introduced in `10e6e8c`.
- Original rationale: keep default false (for macOS concerns) but enable bundling in Linux CI.

### D. Linux custom retag repair command

- Current: `repair-wheel-command = "python -m wheel tags --platform-tag manylinux_2_34_$(uname -m) {wheel} && cp $(dirname {wheel})/*manylinux*.whl {dest_dir}/"`.
- Provenance: `10e6e8c` replaced previous script-based repair from `6076d81`.
- Original rationale: bypass `auditwheel repair` when Mojo runtime libs / ABI checks blocked standard repair.

### E. macOS repair override

- Current: `[tool.cibuildwheel.macos] repair-wheel-command = ""`.
- Provenance: first added in `6e56742`, retained in later commits.
- Original rationale: avoid delocate failures from earlier hatch-mojo/macOS runtime-linking limitations.

### F. Default hatch-mojo bundling policy

- Current: `[tool.hatch.build.targets.wheel.hooks.mojo] bundle-libs = false`.
- Provenance: introduced in `10e6e8c` with issue #7-era comment.
- Original rationale: avoid macOS bundling path until upstream fixes landed.

## 2) Upstream parity map (`../hatch-mojo` v0.1.5)

Sources used: `README.md`, `hatch_mojo/runtime.py`, commit history (`f6b1514`, `d61bafe`).

### Relevant upstream updates

- `f6b1514` (`v0.1.4`): platform-aware sentinel for modular lib discovery.
- `d61bafe` (`v0.1.5`): stronger runtime patching for macOS dylibs, residual RPATH stripping, duplicate guards, plus runtime license notice handling.
- README now states:
  - `bundle-libs = true` works on Linux and macOS.
  - macOS can use normal delocate-based repair command.
  - Linux may still require retag strategy in environments where standard auditwheel policy rejects Mojo-linked wheels.

## 3) Keep/remove decision matrix

| Item | Decision | Confidence | Risk if changed now | Fallback |
|---|---|---|---|---|
| A. `hatch-mojo>=0.1.4` floor | **Change to `>=0.1.5`** | High | Low | Revert floor to `>=0.1.4` if regressions found |
| B. Linux `MODULAR_LIB_DIR` copy staging | **Candidate remove** after validating v0.1.5 autodiscovery in cibuildwheel | Medium | Medium (Linux build break if autodiscovery still fails in manylinux) | Keep env+copy until CI evidence proves safe removal |
| C. Linux `sed` toggle for `bundle-libs` | **Remove** and set canonical config in TOML | High | Low | Restore Linux-only toggle if cross-platform policy cannot be unified |
| D. Linux custom retag repair command | **Keep for now** | Medium | Medium (PyPI tag/repair regressions) | Revisit once auditwheel path is demonstrated stable with Mojo constraints |
| E. macOS `repair-wheel-command = ""` | **Remove empty override**; use standard/default delocate path | High | Medium (if unresolved dylib issues remain in real wheel builds) | Reinstate temporary override with explicit issue reference |
| F. Default `bundle-libs = false` | **Change to `true`** (single source of truth) | High | Medium (wheel size increase, packaging behavior change) | Scope bundling by platform only if evidence requires split policy |

## 4) Chapter 2 change contract

Chapter 2 (`hatch-mojo-ci-realignment_20260301`) should target these exact files:

- `pyproject.toml`
- `.github/workflows/ci.yml` (only if environment/behavior coupling requires workflow edits)
- `.github/workflows/publish.yml` (only if environment/behavior coupling requires workflow edits)
- `src/py/tests/test_mojo_namespace_layout.py`

Expected delta contract:

1. Update hatch-mojo minimum to `>=0.1.5`.
2. Replace toggle-based bundling (`bundle-libs = false` + Linux `sed`) with explicit intended steady-state policy.
3. Remove macOS empty repair override and align with standard/default repair behavior.
4. Evaluate Linux `MODULAR_LIB_DIR` staging removal; if retained, document why and add explicit exit criteria in comments/docs.
5. Keep Linux retag repair unless CI evidence confirms standard repair viability with Mojo constraints.
6. Update config contract test expectations in `test_mojo_namespace_layout.py` to match final policy.

Validation evidence expected from Chapter 2:

- Linux and macOS wheel build outcomes from CI (`cibuildwheel` jobs).
- Explicit verification of resulting wheel tags and runtime linkage behavior.
- No workaround comments that claim upstream issue status contradicted by current `hatch-mojo` releases.

