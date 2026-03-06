# Master PRD: Production Release 0.1.0

## Context

mogemma must ship `v0.1.0` as a production-ready release now, not as an alpha staging release. The release must be technically defensible: real inference behavior, zero-warning quality gates, seamless model download in the default install, verified packaging and publish workflow, and documentation that only claims behavior proven by code and tests.

**North Star:** A user installs `mogemma`, points to a local path or HF model ID, runs embeddings/text generation successfully with deterministic failure behavior, and can trust that docs and release claims map to tested implementation.

## Why This PRD Exists

Recent work improved architecture and docs, but first-release risk still spans multiple domains:
1. Mojo core behavior currently contains placeholder outputs in `src/mo/core.mojo`.
2. Download UX still depends on optional extra semantics while release target requires no-config default download.
3. Local `make lint` semantics still allow soft failures in `Makefile`.
4. Distribution pipeline and release runbook require final production-grade validation.
5. Feature claims and examples require final evidence mapping for release trust.
6. Performance baselines and regression thresholds are not yet formalized for release gating.

## Roadmap

### Chapter 1: `core-inference-parity_20260222`
**Mojo Core Production Parity**  
Replace placeholder core behavior with production semantics and strict contracts for `init_model`, `step`, and `generate_embeddings`.

### Chapter 2: `python-runtime-contracts_20260222`
**Python Runtime Contract Hardening**  
Stabilize wrapper behavior for sync/async generation and embeddings with strict validation and predictable error taxonomy.

### Chapter 3: `seamless-model-delivery_20260222`
**Default Zero-Config Model Delivery**  
Make `huggingface-hub` default and ensure local/cache/download/offline flows are explicit and deterministic.

### Chapter 4: `quality-gates-zero-warning_20260222`
**Zero-Warning Quality Gates**  
Enforce strict lint/type/test behavior locally and in CI with minimal suppressions.

### Chapter 5: `distribution-release-ops_20260222`
**Distribution and Release Operations**  
Validate artifacts, build/test matrix, and publish workflow; establish release checklist and go/no-go gate.

### Chapter 6: `docs-claims-validation_20260222`
**Documentation and Claim Verification**  
Audit and align README/docstrings/examples with real behavior, remove marketing language, and keep wording concise.

### Chapter 7: `performance-baselines_20260222`
**Performance Baselines and Guardrails**  
Create repeatable benchmark baselines and release thresholds for embedding and generation paths.

## Global Constraints

1. Planning and implementation must preserve Python/Mojo architecture boundaries in `src/py/mogemma` and `src/mo`.
2. No unsupported claims in docs, release notes, or README.
3. `make lint` and CI quality jobs must fail on any warning/error for release gating.
4. Default install must support model download without extra user dependency configuration.
5. Error behavior must be explicit and non-silent across missing core/dependency/runtime paths.
6. Every chapter must include explicit validation commands and objective exit criteria.
7. Release publication is blocked until all chapters satisfy validation criteria.

## Chapter Dependency Order

1. `core-inference-parity_20260222`
2. `python-runtime-contracts_20260222`
3. `seamless-model-delivery_20260222`
4. `quality-gates-zero-warning_20260222`
5. `distribution-release-ops_20260222`
6. `docs-claims-validation_20260222`
7. `performance-baselines_20260222`

## Beads Mapping

- Master Epic: `mogemma-bp0`
- Chapter Epics:
  - `core-inference-parity_20260222` -> `mogemma-bp0.1`
  - `python-runtime-contracts_20260222` -> `mogemma-bp0.2`
  - `seamless-model-delivery_20260222` -> `mogemma-bp0.3`
  - `quality-gates-zero-warning_20260222` -> `mogemma-bp0.4`
  - `distribution-release-ops_20260222` -> `mogemma-bp0.5`
  - `docs-claims-validation_20260222` -> `mogemma-bp0.6`
  - `performance-baselines_20260222` -> `mogemma-bp0.7`
