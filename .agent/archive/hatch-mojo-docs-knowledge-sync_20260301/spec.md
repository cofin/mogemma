# Flow: hatch-mojo-docs-knowledge-sync_20260301

## Specification

### Goal

Ensure all operator-facing and agent-facing docs accurately describe the finalized hatch-mojo integration behavior after Chapter 2 changes.

### Requirements

1. Update release runbook guidance for wheel repair/bundling semantics that changed in Chapter 2.
2. Update `.agent/knowledge` notes so historical constraints are clearly marked as superseded or still active with rationale.
3. Keep `.agent/patterns.md` aligned with current build architecture language.
4. Preserve traceability back to Chapter 1 evidence and Chapter 2 implementation decisions.

### Candidate Files

- `docs/release-runbook.md`
- `.agent/knowledge/pypi-cibuildwheel.md`
- `.agent/knowledge/cleanup-and-migrate.md`
- `.agent/patterns.md`

## Implementation Plan

### Phase 1: Update release/operator docs

- [x] 1.1 Revise release runbook steps and caveats to match final CI/build behavior.
- [x] 1.2 Remove stale references to workaround-only pathways that are no longer applicable.

### Phase 2: Update agent knowledge base

- [x] 2.1 Update `.agent/knowledge/pypi-cibuildwheel.md` with current status and constraints.
- [x] 2.2 Add a short “what changed and why” section tied to this PRD outcomes.
- [x] 2.3 Sync any impacted patterns in `.agent/patterns.md`.

### Phase 3: Consistency verification

- [x] 3.1 Verify docs and knowledge entries agree with final `pyproject.toml` and workflow state.
- [x] 3.2 Capture a concise handoff summary for future flow recovery.
