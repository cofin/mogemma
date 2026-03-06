# Learnings: Pure Mojo Transformer

## [2026-02-25] - GQA, MLP, and decoder assembly

- **Implemented:** Full GQA, MLP block, decoder layer loop, embedding pooling.
- **Learnings:**
  - GQA head dimension mapping must be precise (multiple Q heads per KV head).
  - Struct-based weight hierarchy should match model architecture.
  - Two pooling strategies (last-token, mean) — choice affects embedding quality.
  - Attention score scaling by `sqrt(d)` must be exact.
