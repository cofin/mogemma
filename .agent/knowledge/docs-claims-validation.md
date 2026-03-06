# Knowledge: Documentation & Claims Validation

> Flow: docs-claims-validation_20260222 | Archived 2026-02-22

## Patterns

- **Claims-to-evidence mapping:** Every claim must have explicit code path + test evidence.
- **Language precision:** Replace superlatives and marketing wording with concise technical statements.
- **Example verifiability:** Examples must run against real API and ideally have corresponding tests.
- **Late-phase docs:** Keep documentation phase late in release sequence; revalidate after implementation changes.

## Gotchas

- Docs drift after late implementation changes — require final revalidation pass.
- Marketing-style performance claims are unsubstantiated — strip superlatives, reference only measured baselines.
- Example code rot is common — tie examples to tests or CI checks.
