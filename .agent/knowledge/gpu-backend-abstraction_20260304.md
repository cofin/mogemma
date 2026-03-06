# Knowledge: gpu-backend-abstraction_20260304

**Completed:** 2026-03-06
**Archived:** 2026-03-06

## Topics
gpu, backend-abstraction, dispatch, mojo, cpu_mojo, gpu_mojo, adapter-pattern, Python-Mojo-boundary

## Learnings
- 2026-03-05: Keep backend dispatch indirection in Python (`model.py`) via tiny resolver hooks (`_resolve_generation_backend`, `_resolve_embedding_backend`) so tests can pin routing behavior without importing Mojo internals.
- 2026-03-05: A single `CPUCoreBackend` adapter with canonical backend IDs (`cpu_mojo`, `gpu_mojo` aliases) gives a low-risk seam for future GPU backend insertion while preserving current `_core` runtime behavior.
- 2026-03-05: Treat `gpu:N` as a canonical `gpu_mojo` backend resolution case in Python-level parsing so device-index syntax can be accepted early without coupling chapter-1 backend ID logic to concrete GPU runtime implementation.
- 2026-03-05: Record early backend-default decisions in a tracked ADR (`docs/architecture-decisions.md`) so later runtime/device changes have explicit non-MAX and compatibility constraints to follow.

## Summary
Defined backend interface and execution abstraction (cpu_mojo, gpu_mojo, optional adapters) with clear boundaries between Python API, Mojo core runtime state, and kernel dispatch. Introduced resolver hooks in Python for testable backend routing and a CPUCoreBackend adapter as the initial seam for future GPU insertion.
