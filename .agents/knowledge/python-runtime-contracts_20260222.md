# Learnings: Python Runtime Contracts

## [2026-02-22] - Contract hardening

- **Implemented:** Hardened sync/async generation, standardized error taxonomy, deterministic sampling.
- **Learnings:**
  - Document explicit contracts for every public method including edge cases.
  - Zero-temperature requires deterministic behavior; stochastic paths need branch coverage.
  - Lazy tokenizer init introduces non-deterministic failures — validate dependency state upfront.
  - Silent fallbacks mask real issues — eliminate all implicit fallbacks.
