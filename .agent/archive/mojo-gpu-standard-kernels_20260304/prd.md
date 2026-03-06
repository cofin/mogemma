# PRD: Mojo GPU Standard Gemma Kernels

## Context

After foundation work, the standard Gemma path can be accelerated by introducing GPU kernels/dispatch and device-resident cache management.

**North Star:** ship CUDA-first acceleration for standard path with explicit correctness and performance gates.

## Roadmap

### Chapter 1: `gpu-standard-kernels_20260304`

Kernelize or GPU-dispatch standard attention/MLP/norm/rope operations.

### Chapter 2: `gpu-kv-cache-device-memory_20260304`

Move KV cache and critical scratch paths to device memory with defined transfer policy.

### Chapter 3: `gpu-standard-correctness-gates_20260304`

Add deterministic parity and perf gates for standard GPU path.

## Global Constraints

- Depends on GPU foundation PRD completion.
- CPU path remains functional and testable.
- Quantitative perf gates required before enabling by default.
