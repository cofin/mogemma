# Knowledge: Core Embedding Padding

> Flow: core-embedding-padding | Archived 2026-02-24

## Key Changes

Migrated padding logic from Python to Mojo backend. `generate_embeddings` now accepts jagged token arrays with internal padding/masking.

## Patterns

- **Jagged array handling:** List of 1D arrays or flat array with length offsets.
- **Backend padding:** Dynamic padding to max sequence length computed inside Mojo for performance.

## Gotchas

- `numpy.asarray` on jagged lists fails with "inhomogeneous shape" error if padding is disabled.
- Tokenizer padding was previously a no-op — must be explicitly disabled when Mojo handles it.
