# Core Inference Parity Learnings

- `init_model` failures in Mojo should fail early with deterministic validation (empty path, missing path, non-directory path) rather than deferring to runtime pipeline errors.
- Python wrappers should not suppress Mojo initialization failures; surfacing explicit `RuntimeError`s preserves actionable diagnostics for consumers.
- `step` and `generate_embeddings` should be treated as contract-bound operations: validate shape/dtype at the Mojo boundary and fail loudly when backend output contracts are not met.
- Parity tests are most reliable when they directly assert boundary contracts (non-placeholder behavior, dimensionality checks, deterministic failure modes) instead of asserting specific model values.
