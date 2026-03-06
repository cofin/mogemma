# Knowledge: Project README

> Flow: project-readme | Archived

## Public API Surface

`SyncGemmaModel`, `AsyncGemmaModel`, `EmbeddingModel`, `GenerationConfig`, `EmbeddingConfig`, `HubManager`.

## Patterns

- **Optional extras:** `mogemma[embed]`, `mogemma[text]`, `mogemma[hub]`, `mogemma[telemetry]`, `mogemma[all]`.
- **Quick start examples** as primary documentation mechanism (not full API docs).
- **Development setup** references Makefile targets (`make install`, `make test`, `make lint`).

## Gotchas

- Code examples must be syntactically valid Python.
- Installation instructions must match actual extras in `pyproject.toml`.
- `EmbeddingModel` requires Mojo `_core` (not available on all platforms).
