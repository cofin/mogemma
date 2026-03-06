# Flow: gpu-ops-runbook_20260304

**Flow ID:** `gpu-ops-runbook_20260304`  
**Status:** `planned`  
**Parent PRD:** `mojo-gpu-packaging-gates_20260304`  
**Parent Context:** PRD4 Chapter 3 (GPU operations runbook, troubleshooting, fallback controls, and release evidence policy).

## Local Status Snapshot

1. Beads is currently unreliable in this workspace, so local spec artifacts are the current source for implementation status.
2. This flow remains unimplemented locally.
3. Upstream GPU device-selection Phase 2 is now implemented locally in `gpu-device-selection-api_20260304`, so runbook work can treat normalized device grammar/runtime-descriptor fields as fixed user-facing contract.
4. No docs, validator, or runbook-command changes have been applied for this flow yet.

## Specification

### Code Analysis Summary

#### Files Examined
- `docs/release-runbook.md` (current release operational workflow)
- `docs/performance-baselines.md` (current perf evidence policy)
- `tools/validate.py` (manual end-to-end validation script)
- `Makefile` (operator-facing quality and benchmark commands)
- `src/py/mogemma/config.py` (public runtime configuration knobs)
- `src/py/mogemma/model.py` (runtime error surfaces and generation flow)
- `src/py/mogemma/hub.py` (model download and conversion failure surfaces)
- `src/py/mogemma/telemetry.py` (observability baseline)
- `src/mo/mogemma/core.mojo` (engine metadata and runtime state)
- `.github/workflows/ci.yml` and `.github/workflows/publish.yml` (operational release path)

#### Key Findings
- Existing release runbook is distribution-focused and CPU-oriented; it does not define GPU deployment triage, fallback controls, or incident response playbooks (`docs/release-runbook.md:5-14`, `docs/release-runbook.md:58-65`).
- Performance baseline guidance is generic and CPU-centered; it lacks GPU-specific diagnostics, hardware normalization, or backend/device distinctions (`docs/performance-baselines.md:7-17`, `docs/performance-baselines.md:61-67`).
- Validation script checks for `_core.so` and runs semantic checks, but it does not provide GPU-specific preflight/capability diagnostics (`tools/validate.py:103-107`, `tools/validate.py:47-75`).
- Public configs expose a `device` field but no operationally documented behavior/guarantees yet, increasing risk of operator confusion (`src/py/mogemma/config.py:20-21`, `src/py/mogemma/config.py:54-55`).
- Runtime errors for core availability and init failures are present but not currently mapped into an operator troubleshooting decision tree (`src/py/mogemma/model.py:82-100`, `src/py/mogemma/model.py:334-339`).
- Hub download path has explicit GCS and model-not-found error classes that should be surfaced in runbook troubleshooting sections (`src/py/mogemma/hub.py:82-87`, `src/py/mogemma/hub.py:109-111`, `src/py/mogemma/hub.py:123-126`).
- Telemetry support is optional and currently minimal; runbook needs explicit expectations for observability when telemetry deps are absent (`src/py/mogemma/telemetry.py:21-26`).
- Core runtime exposes `engine`, `arch`, and cache metadata that can be leveraged in operational diagnostics if documented (`src/mo/mogemma/core.mojo:281-285`, `src/mo/mogemma/core.mojo:350-364`).

### Relevant Patterns
- Keep claims factual and deterministic; use deterministic validation settings when documenting verification procedures (`.agent/patterns.md`).
- Preserve hybrid Mojo-Python boundary discipline: runbook must separate Python API symptoms from Mojo core/backend symptoms (`.agent/patterns.md`).
- Capture reusable operational lessons into patterns after flow completion to reduce repeated incident cost (`.agent/patterns.md`).
- Performance gotcha: runtime rebuild overhead can masquerade as infrastructure issues; troubleshooting must include runtime-state checks (`.agent/patterns.md`).

### Requirements

#### Functional Requirements
- Define a GPU operations runbook structure covering preflight checks, deployment checks, incident triage, fallback controls, and release evidence.
- Define troubleshooting matrices for at least: capability mismatch, runtime init failure, benchmark regression, and model download failure.
- Define explicit fallback procedures (CPU fallback conditions, escalation, rollback criteria) tied to measurable signals.
- Define operator command catalog for validation and diagnostics (local + CI contexts), including expected outputs and failure interpretation.
- Define release evidence checklist for GPU readiness and operational sign-off.
- Define the mandatory runbook sections and heading contract for `docs/release-runbook.md` and `docs/performance-baselines.md`, so future doc validation can assert their continued presence.
- Define a machine-readable command catalog/evidence manifest for each operator action with, at minimum, `command`, `purpose`, `success_signal`, `failure_class`, `artifact_output`, and `escalation_owner`.
- Define ownership and escalation rules for every incident class, including when the path is operator-mitigatable versus immediate stop-ship.

#### Non-Functional Requirements
- Runbook steps must be executable by on-call engineers without code context switching and with minimal ambiguity.
- Operational guidance must remain aligned with CI/release automation to avoid “docs drift.”
- Troubleshooting paths must prioritize fast mitigation first, then root-cause deep dive.
- Documentation updates must be testable/validatable where possible (command/schema/section presence checks).
- Every documented symptom must map to at least one deterministic validation command and one clear next action.
- Runbook commands must be copy/paste-safe, state offline/network assumptions explicitly, and avoid requiring source inspection to choose the next step.
- Operational checks that inspect runtime/backend state must happen at init or explicit validation time, never inside the token-step hot path.

#### Required Deliverables
- A required-section matrix for `docs/release-runbook.md` and `docs/performance-baselines.md`, including which sections are release-blocking when missing.
- A command/evidence manifest that release automation or doc validators can check for completeness.
- A rehearsal rubric covering time-to-first-mitigation, command success rate, rollback clarity, and evidence completeness.

#### Risks
- GPU operational guidance can go stale quickly if runtime semantics evolve without matching doc updates.
- Ambiguous fallback criteria may cause delayed mitigation during incidents.
- Missing observability dependencies can reduce incident diagnosability unless explicitly covered.
- Cross-platform GPU differences (driver/runtime/tooling) may require segmented runbook branches.

## Implementation Plan

### Phase 1: Incident Taxonomy and Control Surface Definition
- [x] 1.1 Define primary GPU incident classes and map each to severity, detection signals, immediate actions, stop-ship status, and escalation owner.
- [x] 1.2 Inventory runtime controls and exposed diagnostics from `src/py/mogemma/config.py`, `src/py/mogemma/model.py`, `src/py/mogemma/hub.py`, `src/py/mogemma/telemetry.py`, `tools/validate.py`, and runtime metadata surfaces.
- [x] 1.3 Define fallback policy matrix keyed by requested device, detected capability, deployment stage, and observed failure class (auto/manual CPU fallback, block, rollback).
- [x] 1.4 Checkpoint: review incident/control taxonomy with implementation owners.

### Phase 2: Runbook Authoring Plan
- [x] 2.1 Plan updates to `docs/release-runbook.md` for required GPU sections: preflight, deployment verification, fallback/rollback, incident triage, evidence collection, and owner/escalation tables.
- [x] 2.2 Plan updates to `docs/performance-baselines.md` for backend-aware evidence capture, runner normalization, and regression interpretation.
- [x] 2.3 Plan command reference updates (`tools/validate.py` and `Makefile` usage) with expected success/failure snippets and the artifacts each command must emit or update.
- [x] 2.4 Checkpoint: validate that each documented symptom has at least one deterministic verification command.

### Phase 3: Verification and Release Evidence Integration
- [x] 3.1 Define how runbook evidence integrates into release go/no-go checklist and artifact collection, including a manifest with timestamps, host/backend metadata, benchmark artifacts, validation output, and fallback result.
- [x] 3.2 Define CI/doc validation checks that prevent shipping stale or incomplete runbook sections, command references, or evidence-manifest fields.
- [x] 3.3 Define dry-run protocol for simulated GPU incident response and fallback execution on both a GPU-capable host and a CPU-only control environment.
- [x] 3.4 Checkpoint: confirm release sign-off criteria include runbook completeness and rehearsal evidence.

### Phase 4: Manual Operational Verification Plan
- [x] 4.1 Define manual rehearsal scenarios (init failure, perf regression, download failure, fallback/rollback).
- [x] 4.2 Define runbook acceptance rubric (time-to-first-mitigation, command success rate, escalation clarity, rollback clarity, evidence completeness).
- [x] 4.3 Manual verification task: execute at least one full incident rehearsal using the runbook draft plus one explicit CPU-fallback drill, recording timings, outcomes, emitted artifacts, and remaining gaps.
- [x] 4.4 Checkpoint: lock final runbook section ownership and update cadence.

## TDD-Oriented Task Checklist
- [ ] Add failing doc-contract tests/validators for required runbook sections and mandatory command references.
- [ ] Add failing checks for evidence checklist completeness, including GPU-specific artifacts, fallback outcomes, and ownership metadata.
- [ ] Add failing tests for any new validation-script or benchmark-output contracts referenced by the runbook.
- [ ] Implement the minimal doc and validator changes needed to satisfy the failing checks while keeping commands aligned with actual automation.
- [ ] Re-run docs/validation tests and ensure existing release checks remain green.
- [ ] Refactor runbook language for concise operator execution while preserving explicit decision criteria.
- [ ] Add regression guard checks so future workflow/config changes cannot silently invalidate runbook commands, section headers, or evidence-manifest fields.
