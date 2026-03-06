# Flow: distribution-release-ops_20260222

## Specification

### Goal

Finalize packaging, artifact validation, and release operations so `v0.1.0` can be published safely with reproducible build and publish steps.

### Code Analysis Summary

1. `.github/workflows/publish.yml` builds source and platform wheels, then publishes to PyPI on release event.
2. `.github/workflows/test-build.yml` provides partial workflow coverage focused on build configuration changes.
3. `tools/hatch_build.py` compiles Mojo core into wheel build path and is release-critical.
4. `pyproject.toml` contains build metadata, extras, and versioning settings for `bump-my-version`.
5. Release process needs explicit operational checklist and final go/no-go policy for first production release.

### Requirements

1. Artifact matrix for first release must be documented and validated.
2. Wheel and sdist artifacts must install and import cleanly in isolated environments.
3. Publish workflow must be verified end-to-end before production publication.
4. Versioning/tagging/release steps must be explicit and deterministic.
5. Rollback/failure handling must be included in release runbook.

### Non-goals

1. Introducing additional package indexes beyond required release scope.
2. Multi-project release orchestration.
3. Major restructuring of CI architecture beyond release necessities.

## Implementation Plan

### Phase 1: Packaging and Metadata Verification

- [x] 1.1 Validate `pyproject.toml` metadata for classifiers, dependencies, and package layout correctness.
- [x] 1.2 Verify wheel includes required runtime files (`_core.so`, `py.typed`, package modules).
- [x] 1.3 Verify sdist includes required build and source files.
- [x] 1.4 Confirm version metadata and bump configuration are release-ready.

### Phase 2: Build Matrix Validation

- [x] 2.1 Confirm target support matrix for first release (OS and Python versions) and document it explicitly.
- [x] 2.2 Align `.github/workflows/publish.yml` and `.github/workflows/test-build.yml` with declared matrix.
- [x] 2.3 Ensure matrix is realistic for Mojo toolchain support and remove unsupported claims.
- [x] 2.4 Add artifact integrity checks (package listing/signature/hash reporting) where useful.

### Phase 3: Artifact Install and Smoke Verification

- [x] 3.1 Add/expand isolated-environment smoke tests for built wheels.
- [x] 3.2 Verify import/runtime smoke checks for both embedding and generation entry points.
- [x] 3.3 Verify package works with default install assumptions and required dependencies.
- [x] 3.4 Validate test commands used in workflow are deterministic.

### Phase 4: Publish Workflow Dry-Run and Release Runbook

- [x] 4.1 Create release runbook with explicit command order, prerequisites, and expected outputs.
- [x] 4.2 Add dry-run procedure (non-production publish rehearsal) and capture evidence.
- [x] 4.3 Define rollback/failure handling for failed publish/build scenarios.
- [x] 4.4 Define required approvals/checks before invoking production publish.

### Phase 5: Final Release Gate

- [x] 5.1 Define a single go/no-go checklist covering quality, docs, perf, packaging, and runtime verification.
- [x] 5.2 Validate checklist completion with evidence links from each chapter.
- [x] 5.3 Record final release recommendation and blockers (if any) in Beads notes.

## Test Plan

1. `uv build --sdist`
2. `uv build --wheel`
3. `uv run python -c "import mogemma; print('import ok')"`
4. Workflow dry-run commands from runbook
5. `make test`
6. `make lint`

## Validation Criteria

1. Artifacts build successfully and install in isolated environments.
2. Declared support matrix is tested and documented.
3. Publish workflow is verified via dry-run procedure.
4. Release runbook and rollback plan are complete and actionable.
5. Final go/no-go checklist is fully satisfiable before release.

## Risks and Mitigations

1. Risk: Platform-specific build failures close to release.
   Mitigation: Validate matrix early and keep fallback documented.
2. Risk: Packaging metadata drift causes install errors.
   Mitigation: Add artifact inspection and isolated install tests as hard gates.
