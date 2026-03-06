# Knowledge: Pure Mojo Transformer

> Flow: pure-mojo-transformer | Archived 2026-02-25

## Key Changes

Built full Grouped Query Attention (GQA) mechanism, MLP block, decoder layer loop, and embedding pooling — all in Mojo. Struct-based weight management matching model architecture hierarchy.

## Patterns

- **Struct-based weights:** Nested hierarchy (`GemmaModelWeights` > `DecoderLayerWeights`) matching model architecture.
- **GQA:** Multiple query heads share single KV head — careful head dimension mapping required.
- **Two pooling strategies:** Last-token and mean-pooling for embedding extraction.
- **Residual connections + RMSNorm** between attention and MLP blocks.

## Gotchas

- GQA head dimension mapping must be precise (multiple Q heads per KV head).
- Attention score scaling by `sqrt(d)` must be exact.
- Pooling strategy choice affects downstream embedding quality.
