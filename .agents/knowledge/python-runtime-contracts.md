# Knowledge: Python Runtime Contracts

> Flow: python-runtime-contracts_20260222 | Archived 2026-02-22

## Key Changes

Hardened sync/async generation with deterministic sampling. Standardized error taxonomy across all paths. Added contract-level assertions for sampling branches.

## Patterns

- **Contract matrix:** Document explicit contracts for every public method including edge cases.
- **Numeric stability:** Zero-temperature paths require deterministic behavior; stochastic paths need branch coverage.
- **Error consistency:** Normalize error messages across missing tokenizer, missing core, and runtime failure paths.

## Gotchas

- Sampling changes alter output expectations silently — use deterministic tests for zero-temp.
- Async cancellation semantics are environment-sensitive.
- Lazy tokenizer initialization can introduce non-deterministic failures — validate dependency state upfront.
- Silent fallback behavior in generation paths masks real issues — eliminate all implicit fallbacks.
