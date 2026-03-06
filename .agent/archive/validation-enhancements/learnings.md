# Learnings: Validation Enhancements

## [2026-02-25] - Semantic validation gates

- **Implemented:** Semantic prompt suite with factual contracts, deterministic decoding, failure taxonomy.
- **Learnings:**
  - Deterministic decoding (`temperature=0`) required for reliable factual validation.
  - Single prompt insufficient for catching semantic degradation — use 3-5 diverse prompts.
  - Failure taxonomy: model_resolution, runtime_inference, semantic_validation.
  - Non-deterministic sampling masks quality regressions.
