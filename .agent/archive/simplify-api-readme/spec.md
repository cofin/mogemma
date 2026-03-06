# Flow: simplify-api-readme

## Specification

### Code Analysis

**Constructor signatures today:**
- `SyncGemmaModel(config: GenerationConfig, tokenizer=None)` — config required
- `AsyncGemmaModel(config: GenerationConfig)` — config required
- `EmbeddingModel(config: EmbeddingConfig, tokenizer=None)` — config required

**Defaults already exist:** `GenerationConfig()` works with all defaults (model=gemma3-270m-it, max_tokens=128, temp=1.0). Same for `EmbeddingConfig()`. The config classes have sensible defaults — they just aren't leveraged by model constructors.

**README:** Currently shows `config = GenerationConfig(max_tokens=64)` which overrides the default (128) for no clear reason, adding confusion.

### Requirements

1. All three model classes accept an optional first argument: `str | Config | None`
   - `None` / omitted → create default config
   - `str` → create config with that model path
   - Config object → use as-is (backward compatible)
2. README quickstart shows zero-config usage
3. Existing `SyncGemmaModel(config)` calls continue working (no breaking change)

### Affected Files

| File | Change |
|------|--------|
| `src/py/mogemma/model.py` | Update `__init__` signatures for all 3 classes |
| `src/py/mogemma/__init__.py` | No change needed (exports already correct) |
| `README.md` | Simplify quickstart examples |
| `src/py/tests/test_gemma_model.py` | Add tests for new constructor forms |
| `src/py/tests/test_embeddings.py` | Add tests for new constructor forms |

## Implementation Plan

### Phase 1: Model constructor changes

- [x] 1.1 Update `SyncGemmaModel.__init__` to accept `str | GenerationConfig | None = None`
- [x] 1.2 Update `AsyncGemmaModel.__init__` to accept `str | GenerationConfig | None = None`
- [x] 1.3 Update `EmbeddingModel.__init__` to accept `str | EmbeddingConfig | None = None`
- [x] 1.4 Add tests for: no-arg, string-arg, config-arg constructors

### Phase 2: README simplification

- [x] 2.1 Rewrite quickstart: sync example with no config
- [x] 2.2 Rewrite quickstart: async example with no config
- [x] 2.3 Rewrite quickstart: embedding example with no config
- [x] 2.4 Show model variant selection with string arg
