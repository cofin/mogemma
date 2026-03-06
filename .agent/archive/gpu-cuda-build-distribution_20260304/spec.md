# Flow: gpu-cuda-build-distribution_20260304

**Flow ID:** `gpu-cuda-build-distribution_20260304`  
**Status:** `planned`  
**Parent PRD:** `mojo-gpu-packaging-gates_20260304`  
**Parent Context:** PRD4 Chapter 1 (CUDA build/distribution strategy) aligned to research in `.agent/research/gpu-acceleration-options_20260304/research.md`.

## Local Status Snapshot

1. Beads is currently unreliable in this workspace, so local spec artifacts are the current source for implementation status.
2. This flow remains unimplemented locally.
3. Upstream GPU device-selection Phase 2 is now implemented locally in `gpu-device-selection-api_20260304`, so packaging/install policy can treat normalized device grammar and runtime-descriptor fields as fixed input assumptions.
4. No build/distribution config, CI workflow, or packaging-doc changes have been applied for this flow yet.

## Specification

### Code Analysis Summary

#### Files Examined
- `pyproject.toml` (build backend, wheel hooks, cibuildwheel policy)
- `.github/workflows/ci.yml` (build/test/release-quality jobs)
- `.github/workflows/publish.yml` (release artifact publication workflow)
- `src/py/tests/test_mojo_namespace_layout.py` (packaging/build-target assertions)
- `src/py/mogemma/config.py` (public `device` surface)
- `src/py/mogemma/model.py` (Python-to-core runtime boundary)
- `src/mo/mogemma/core.mojo` (single exported extension/runtime entrypoints)
- `src/mo/mogemma/ops.mojo` (CPU-oriented kernels)
- `src/mo/mogemma/layers.mojo` (CPU attention/MLP loops)
- `.agent/patterns.md` and `.agent/research/gpu-acceleration-options_20260304/research.md`

#### Key Findings
- Current wheel build produces a single extension module `mogemma._core` from `src/mo/mogemma/core.mojo`, with no GPU-specific module split or backend-specific artifact layout (`pyproject.toml:83-89`).
- Linux wheels rely on an in-place `sed` mutation to flip `bundle-libs` for CI builds; this is brittle for introducing CUDA-specific packaging variants (`pyproject.toml:64-66`, `pyproject.toml:80-81`).
- Linux repair retags wheels to manylinux 2.34 instead of using standard auditwheel repair due Mojo runtime symbol constraints, so CUDA distribution policy must preserve this workaround or replace it safely (`pyproject.toml:66-69`).
- CI and publish workflows currently build only OS/arch matrix wheels (Linux/macOS) and have no GPU-specific artifact branch, metadata, or validation steps (`.github/workflows/ci.yml:87-114`, `.github/workflows/publish.yml:33-58`).
- The API already exposes `device` in both generation and embedding configs, but execution path does not dispatch by device yet, so packaging/install checks must be coordinated with runtime semantics (`src/py/mogemma/config.py:20-21`, `src/py/mogemma/config.py:54-55`, `src/py/mogemma/model.py:345-356`).
- Core compute path remains CPU loop oriented with pointer math and per-token allocations, reinforcing the need for explicit CUDA artifact policy before runtime cutover (`src/mo/mogemma/core.mojo:404-408`, `src/mo/mogemma/ops.mojo:68-96`, `src/mo/mogemma/layers.mojo:73-98`).
- Existing packaging tests only validate namespaced source location and manylinux retag command; there is no test scaffold for CUDA wheel policy yet (`src/py/tests/test_mojo_namespace_layout.py:22-45`).

### Relevant Patterns
- Packaging baseline: keep `hatch-mojo>=0.1.5` with `bundle-libs = true` as canonical wheel policy, and document Linux-specific overrides with explicit rationale/exit criteria (`.agent/patterns.md`).
- Contract testing pattern: enforce packaging/runtime contracts via deterministic tests at the Python/Mojo boundary before implementation rollout (`.agent/patterns.md`).
- Source layout pattern: Mojo code stays under `src/mo/mogemma` and Python API under `src/py/mogemma`; build policy must preserve this boundary (`.agent/patterns.md`, `src/py/tests/test_mojo_namespace_layout.py:12-33`).
- Performance gotcha: avoid runtime rebuild overhead in hot paths; packaging choices must not add additional per-token startup overhead (`.agent/patterns.md`, `.agent/research/gpu-acceleration-options_20260304/research.md`).

### Requirements

#### Functional Requirements
- Define supported CUDA distribution strategy for PRD4 Chapter 1, including artifact naming/versioning policy and CPU/GPU compatibility expectations.
- Specify exact `pyproject.toml` changes required for CUDA-capable wheels without regressing current CPU wheel install behavior.
- Define install-time capability validation behavior (driver/runtime check policy, actionable error contracts, and fallback semantics).
- Define CI workflow changes for GPU build validation, including where GPU-specific validation runs and how it is gated.
- Define release evidence required to publish CUDA artifacts (build logs, smoke import, compatibility matrix, and fallback behavior confirmation).
- Define an explicit artifact manifest contract for every published wheel/build candidate with, at minimum, `artifact_id`, `distribution_channel`, `target_platform`, `python_tag`, `backend`, `cuda_runtime`, `glibc_floor`, `runner_class`, `validation_command`, and `fallback_behavior`.
- Define the exact file touch set for the implementation PR so packaging work stays surgical: `pyproject.toml`, `.github/workflows/ci.yml`, `.github/workflows/publish.yml`, `src/py/tests/test_mojo_namespace_layout.py`, `src/py/tests/test_validate_script.py`, and the release docs that describe the resulting policy.
- Define how capability checks are split between import/install time and first backend initialization so CUDA detection never leaks into the token-step hot path.

#### Non-Functional Requirements
- Preserve reproducibility for existing CPU wheel builds across supported Python versions and platforms.
- Keep packaging behavior deterministic and automation-friendly (no manual mutation outside explicit build steps).
- Ensure user-facing failure messages are actionable and tied to concrete remediation steps.
- Maintain compatibility with existing source distribution and trusted publishing release flow.
- Keep PR-time validation bounded by one canonical Linux GPU smoke-build lane plus CPU regression coverage; any wider CUDA/runtime matrix expansion belongs to release-only validation.
- Ensure CUDA capability probing is cached once per process or backend initialization and adds no extra work to per-token generation or embedding loops.
- Require machine-readable build metadata and validation outputs so release evidence can be assembled without manual transcription.

#### Required Deliverables
- A packaging decision table embedded in the spec or linked artifact that lists every supported wheel class, its distribution channel, required runtime, validation command, and fallback rule.
- A manifest/output schema for build artifacts and validation evidence that release automation can archive alongside wheel files.
- A PR-vs-release job split showing which GPU validations are mandatory in `.github/workflows/ci.yml` and which are deferred to `.github/workflows/publish.yml` or dedicated release runners.

#### Risks
- Wheel matrix explosion (OS x arch x Python x CUDA runtime) can exceed CI capacity and release cycle timelines.
- Current Linux `sed`-based bundling toggle may create hidden drift between local and CI build behavior.
- CUDA runtime/library compatibility mismatches may produce install-time failures that are hard to diagnose without explicit checks.
- Lack of runtime device dispatch parity can produce a packaging/runtime contract mismatch if artifact policy lands before backend selection semantics.

## Implementation Plan

### Phase 1: Packaging Contract and Matrix Design
- [x] 1.1 Define artifact strategy as a table with required columns: artifact identifier, channel, wheel naming/tagging rule, backend, CUDA runtime band, glibc floor, runner class, validation command, and fallback semantics.
- [x] 1.2 Produce separate PR and release platform matrices (including Linux x86_64/aarch64, macOS behavior, Python versions) with explicit in/out-of-scope decisions so CI cost is bounded before implementation starts.
- [x] 1.3 Define install-time and first-backend-init validation contracts, including exactly which checks run at each stage, cached-result behavior, and user-visible error/fallback expectations.
- [x] 1.4 Checkpoint: publish signed-off packaging contract in this flow spec before touching build config.

#### Packaging Contract & Matrix (Phase 1 Deliverables)

**Artifact Strategy (1.1)**
| Artifact ID | Channel | Wheel Tag / Naming Rule | Backend | CUDA Runtime | glibc Floor | Runner Class | Validation Command | Fallback Semantics |
|-------------|---------|-------------------------|---------|--------------|-------------|--------------|--------------------|--------------------|
| `cpu-linux-x86_64` | PyPI | `manylinux_2_34_x86_64` | CPU | N/A | 2.34 | `ubuntu-latest` | `python tools/validate.py` | CPU-only execution |
| `cpu-linux-aarch64`| PyPI | `manylinux_2_34_aarch64`| CPU | N/A | 2.34 | `ubuntu-latest` + QEMU | `python tools/validate.py` | CPU-only execution |
| `cpu-macos-arm64` | PyPI | `macosx_11_0_arm64` | CPU | N/A | N/A | `macos-latest` | `python tools/validate.py` | CPU-only execution |
| `cuda-linux-x86_64`| PyPI / Custom | `manylinux_2_34_x86_64` (with `+cu12` local version or custom index) | GPU (CUDA) | >= 12.0 | 2.34 | `ubuntu-latest` | `python tools/validate.py --device gpu` | Fallback to CPU |

**Platform Matrices (1.2)**
*PR Validation Matrix (CI scope bounded):*
- Ubuntu (x86_64): Python 3.12 (CPU tests, standard packaging layout tests)
- MacOS (arm64): Python 3.12 (CPU tests)
- *Note:* GPU builds are mocked/stubbed or skipped in standard PR CI unless a dedicated GPU runner is present to avoid long build times and high CI costs.

*Release Validation Matrix:*
- Linux x86_64 (CPU): Python 3.10, 3.11, 3.12, 3.13, 3.14
- Linux aarch64 (CPU): Python 3.10, 3.11, 3.12, 3.13, 3.14
- MacOS x86_64 / arm64 (CPU): Python 3.10-3.14
- Linux x86_64 (CUDA): Python 3.10-3.14 (built and validated on GPU runner)

**Validation Contracts (1.3)**
- **Install-time:** Wheel install completes regardless of underlying hardware. The wheel contains standard CPU code + GPU paths if compiled. No driver checks during `pip install`.
- **First-backend-init:** When `DeviceSelection` requests `gpu` (via `device="gpu"` or `gpu:0`), `resolve_device_selection` queries a one-time cached capability check (e.g. `gpu_capability_available()`). 
  - If GPU requested but unavailable: throws actionable `RuntimeError` ("Requested device 'gpu' is unavailable on this host.").
  - If CPU requested (or default): bypasses GPU checks entirely.
  - Capability checks MUST NOT occur per-token.



### Phase 2: Build Configuration Plan
- [ ] 2.1 Plan `pyproject.toml` changes for CUDA-aware builds, explicitly calling out whether each edit lands in `[tool.hatch.build.targets.wheel.hooks.mojo]`, `cibuildwheel` config, repair commands, or environment wiring.
- [ ] 2.2 Plan dedicated contract coverage in `src/py/tests/test_mojo_namespace_layout.py` and `src/py/tests/test_validate_script.py` for artifact policy, validation messaging, and CPU-regression protection.
- [ ] 2.3 Define migration strategy away from brittle in-place config mutation, either by removing the `sed` path or isolating it behind a generated, test-covered build overlay with a clear exit criterion.
- [ ] 2.4 Checkpoint: confirm all planned config edits preserve current CPU distribution path.

### Phase 3: CI and Release Pipeline Plan
- [ ] 3.1 Design CI job topology with an explicit split between PR smoke validation in `.github/workflows/ci.yml` and release-matrix validation in `.github/workflows/publish.yml`, including runner type, trigger rules, required artifacts, and failure policy.
- [ ] 3.2 Design publish workflow updates for CUDA artifact inclusion/exclusion, provenance reporting, and manifest attachment so each uploaded artifact can be traced back to a validated runner/runtime pair.
- [ ] 3.3 Define release evidence template covering build logs, wheel hashes, import smoke, capability-check output, compatibility checklist, and CPU fallback validation.
- [ ] 3.4 Checkpoint: verify planned pipeline changes remain compatible with trusted publishing flow.

### Phase 4: Operational Acceptance Plan
- [ ] 4.1 Define explicit manual verification script for CUDA wheel install and runtime smoke checks on a supported GPU host and a CPU-only host, including expected command outputs and failure classification.
- [ ] 4.2 Define rollback criteria and fallback behavior when CUDA wheel validation fails.
- [ ] 4.3 Manual verification task: execute the planned install/smoke matrix on at least one CUDA-capable host and one CPU-only host, recording wheel install result, import smoke, first-init result, fallback result, and artifact manifest linkage in release evidence.
- [ ] 4.4 Checkpoint: confirm go/no-go criteria and ownership for release sign-off.

## TDD-Oriented Task Checklist
- [ ] Add failing packaging tests in `src/py/tests/test_mojo_namespace_layout.py` that encode the expected CUDA artifact policy and preserve the CPU wheel baseline before build config edits.
- [ ] Add failing workflow contract tests (or workflow snapshot checks) for required CUDA build/publish job presence and manifest artifact upload behavior.
- [ ] Add failing validation tests in `src/py/tests/test_validate_script.py` for install-time and first-init capability error contracts.
- [ ] Implement the minimal config/workflow changes required to satisfy the failing tests without widening the matrix beyond the approved PR scope.
- [ ] Re-run full packaging + contract test suite until green.
- [ ] Refactor build config for readability and remove accidental duplication while staying green.
- [ ] Add regression guard tests for the CPU wheel path to prevent CUDA changes from breaking baseline distribution or trusted publishing.
- [ ] Capture final evidence bundle requirements and map each field to an automated or manual verification step.
