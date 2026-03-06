# Knowledge: Validation Enhancements

> Flow: validation-enhancements | Archived 2026-02-25

## Patterns

- **Semantic gating:** Factual contracts (e.g., "What is the capital of France?" -> must contain "Paris").
- **Failure taxonomy:** `model_resolution_failure`, `runtime_inference_failure`, `semantic_validation_failure`.
- **Deterministic decoding:** `temperature=0` for reliable factual validation. Non-deterministic sampling masks quality regressions.

## Gotchas

- Single prompt insufficient for catching semantic degradation — use 3-5 diverse prompts.
- Must distinguish between download, runtime, and quality failures.
