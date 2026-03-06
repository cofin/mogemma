# Learnings: simplify-api-readme

## 2026-03-02

- For constructor simplification, tests should cover all accepted input forms (`None`, `str`, config object) explicitly per model class.
- When default model IDs can trigger remote resolution, constructor tests are more reliable when `HubManager.resolve_model` is monkeypatched to a local fixture path.
- README churn should be avoided when existing examples already match the intended public API contract.

