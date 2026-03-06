# Learnings: Core Embedding Padding

## [2026-02-24] - Padding migration to Mojo

- **Implemented:** Migrated padding from Python to Mojo backend. Jagged array handling.
- **Learnings:**
  - `numpy.asarray` on jagged lists fails with "inhomogeneous shape" error.
  - Tokenizer padding was a no-op — must be explicitly disabled when Mojo handles it.
  - Dynamic padding to max sequence length computed inside Mojo for performance.
