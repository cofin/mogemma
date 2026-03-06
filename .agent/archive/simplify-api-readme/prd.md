# PRD: Simplify API & README

## Context

The current API requires users to create a `GenerationConfig` / `EmbeddingConfig` object before creating a model, even when using all defaults. This adds friction for the simplest use case. The README examples show `max_tokens=64` which is confusing (the actual default is 128) and requires understanding config before getting started.

**North Star:** `model = SyncGemmaModel()` and `model.generate("Hello")` should just work.

## Roadmap

### Chapter 1: `simplify-api-readme`

Make config optional across all three model classes, accept a model ID string as shorthand, and rewrite README examples to be dead simple.

**Scope:**
- `SyncGemmaModel(model_or_config?)` — optional first arg (string model ID or config object)
- `AsyncGemmaModel(model_or_config?)` — same
- `EmbeddingModel(model_or_config?)` — same
- README: keep sync, async, embeddings sections but remove config boilerplate
- Tests: update for new constructor signatures

**Out of scope:**
- No `**kwargs` forwarding to config — use `GenerationConfig`/`EmbeddingConfig` for overrides
- No model registry or enum — IDs are opaque strings resolved by `HubManager`
- No changes to config dataclasses themselves

## Global Constraints

- Maintain backward compatibility: `SyncGemmaModel(config)` must still work
- Type annotations must be correct for pyright/mypy
- Default model remains `gemma3-270m-it`
