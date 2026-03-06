# PRD: Mojo GPU Foundation (non-MAX)

## Context

Current `mogemma` runtime is CPU-only in practice: `device` is not wired to execution and token-step runtime state mixes concerns that will block clean GPU acceleration.

**North Star:** establish backend abstraction, runtime-state refactor, and device API semantics so GPU execution can be added without destabilizing inference correctness.

## Roadmap

### Chapter 1: `gpu-backend-abstraction_20260304`

Define backend interface and dispatch boundaries (`cpu_mojo`, `gpu_mojo`, optional adapters), including lifecycle and error contracts.

### Chapter 2: `gpu-runtime-state-refactor_20260304`

Remove per-token model-wrapper rebuild and separate session/model state from step execution.

### Chapter 3: `gpu-device-selection-api_20260304`

Implement explicit device selection semantics (`cpu|gpu|gpu:N`) and capability/fallback behavior.

## Global Constraints

- No MAX dependency in primary runtime path.
- Keep API backward compatibility where feasible.
- Preserve deterministic CPU behavior while introducing backend interfaces.
