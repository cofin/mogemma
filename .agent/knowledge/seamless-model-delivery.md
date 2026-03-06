# Knowledge: Seamless Model Delivery

> Flow: seamless-model-delivery_20260222 | Archived 2026-02-22

## Key Changes

Migrated `huggingface-hub` from optional extras to base dependencies for zero-config downloads. Hardened resolution order: local path -> cache -> remote download. Made offline/auth failures explicit.

## Patterns

- **Resolution hierarchy:** Local path > cache > remote download. Clear documented precedence prevents ambiguous fallbacks.
- **Dependency policy migration:** Move optional integrations to base when they enable critical user workflows.
- **Dead code audit:** Optional dependency facades become dead code after policy changes — remove unreachable branches.

## Gotchas

- Heuristics for distinguishing model IDs from local paths can misclassify edge cases.
- Dependency migration changes environment reproducibility — refresh lockfile and validate in clean environments.
- Cache directory behavior must be stable and documented.
