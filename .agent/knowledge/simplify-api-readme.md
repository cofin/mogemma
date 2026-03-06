# Knowledge: simplify-api-readme

**Completed:** 2026-03-06
**Archived:** 2026-03-06

## Topics
API design, constructor simplification, README, testing, monkeypatch, model config

## Learnings
- For constructor simplification, tests should cover all accepted input forms (`None`, `str`, config object) explicitly per model class.
- When default model IDs can trigger remote resolution, constructor tests are more reliable when `HubManager.resolve_model` is monkeypatched to a local fixture path.
- README churn should be avoided when existing examples already match the intended public API contract.

## Summary
Simplified model constructors to accept optional config parameters and updated README examples accordingly. Established testing patterns using monkeypatched model resolution to avoid remote dependencies in constructor tests.
