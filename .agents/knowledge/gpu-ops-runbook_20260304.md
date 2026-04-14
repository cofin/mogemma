# Learnings for Flow: gpu-ops-runbook_20260304

## Summary
Completed the following key tasks:
  - 1.1 Define primary GPU incident classes and map each to severity, detection signals, immediate actions, stop-ship status, and escalation owner.
  - 1.2 Inventory runtime controls and exposed diagnostics from `src/py/mogemma/config.py`, `src/py/mogemma/model.py`, `src/py/mogemma/hub.py`, `src/py/mogemma/telemetry.py`, `tools/validate.py`, and runtime metadata surfaces.
  - 1.3 Define fallback policy matrix keyed by requested device, detected capability, deployment stage, and observed failure class (auto/manual CPU fallback, block, rollback).
  - 1.4 Checkpoint: review incident/control taxonomy with implementation owners.
  - 2.1 Plan updates to `docs/release-runbook.md` for required GPU sections: preflight, deployment verification, fallback/rollback, incident triage, evidence collection, and owner/escalation tables.
  - ... and 11 more tasks.

## Patterns & Gotchas
- **Architecture**: Ensure components are fully aligned with project conventions outlined in `.agent/patterns.md`.
- **Validation**: Rely on empirical testing and standard parity gates before finalizing flows.
