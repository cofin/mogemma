# Knowledge: Core Inference Parity

> Flow: core-inference-parity_20260222 | Archived 2026-02-22

## Key Changes

Replaced placeholder Mojo core behavior (`np.zeros`, `np.random.rand`) with production-grade inference. Locked core contracts: `init_model`, `step`, `generate_embeddings` with explicit dtype/shape guarantees.

## Patterns

- **Contract-first design:** Define and lock core API contracts before implementation changes.
- **Phased approach:** Define contracts -> Replace behavior -> Align Python wrappers -> Add parity tests -> Validate.
- **Separation of concerns:** Core implementation decoupled from Python wrappers via explicit contracts.

## Gotchas

- Core API drift breaks Python wrappers if contracts aren't locked first.
- Non-deterministic runtime behavior causes flaky tests — use deterministic fixtures.
- Placeholder stubs that pass initial tests can silently masquerade as valid until replaced.
