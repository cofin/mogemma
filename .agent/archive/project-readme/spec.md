# Flow: Project README

## Specification

### Code Analysis Summary

**Public API (from `__init__.py`):**
- `SyncGemmaModel` — synchronous text generation with streaming
- `AsyncGemmaModel` — async wrapper using `asyncio.to_thread`
- `EmbeddingModel` — embedding generation (requires Mojo `_core`)
- `GenerationConfig` — config dataclass (model_path, temperature, top_k, top_p, etc.)
- `EmbeddingConfig` — config dataclass (model_path, max_sequence_length, batch_size)
- `HubManager` — resolves model paths (local, cache, HF Hub)

**Optional extras:**
- `mogemma[embed]` — numpy (for embeddings)
- `mogemma[text]` — numpy + tokenizers (for text generation)
- `mogemma[hub]` — huggingface-hub (for model downloading)
- `mogemma[telemetry]` — opentelemetry-api (for tracing)
- `mogemma[all]` — everything

**Project identity:** Python/Mojo hybrid for Google Gemma 3. Lightweight embeddings + slim LLM serving. The Mojo backend provides native acceleration via MAX Engine.

### Requirements

1. Create `README.md` at project root
2. Keep it brief and friendly — not a full docs site
3. Sections: overview, features, installation, quick start (embeddings + generation + async), optional extras, development setup
4. No badges yet (CI isn't set up, PyPI isn't published)

### Non-goals
- No full API reference (that's for docs)
- No contributing guide (separate file later)

---

## Implementation Plan

### Phase 1: Write README
- [x] 1.1 Create `README.md` with project overview and feature highlights [65a010d]
- [x] 1.2 Add installation section with optional extras [65a010d]
- [x] 1.3 Add quick start code examples (embeddings, text generation, async streaming) [65a010d]
- [x] 1.4 Add development setup section (make install, make test, make lint) [65a010d]

### Validation Criteria
- README renders correctly as GitHub markdown
- Code examples are syntactically valid Python
- Installation instructions match actual extras in pyproject.toml
