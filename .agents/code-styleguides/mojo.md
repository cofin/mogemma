# Mojo Style Guide

House style for Mojo code in `src/mo/mogemma/` and `src/mo/tests/`. Mojo is
young and evolving — when this guide disagrees with the stdlib's latest
idioms, the stdlib wins. Update this file when you discover a better pattern.

## Tooling

- **Formatter**: Mojo's built-in `mojo format`. Line length 120.
- **Test harness**: `mojo test` over files matching `test_*.mojo` in `src/mo/tests/`.
- Run `make lint` to check formatting; `make test` includes Mojo tests.

## File layout

```mojo
# SPDX license line (if any)

from std.memory import UnsafePointer          # stdlib first
from std.math import sqrt, erf, tanh
from std.sys import has_accelerator

from mogemma.gpu_context import (             # first-party grouped
    WeightStage,
    GPUContext,
    upload_layer_weights,
)
from mogemma.model import (
    LayerWeights,
    ModelWeights,
    TensorInfo,
)
from mogemma.ops import (
    ComputeBackend,
    CPUBackend,
    vec_mat_mul,
    rms_norm,
)


alias LAYER_TYPE_SLIDING = 0                   # module-level aliases
alias LAYER_TYPE_FULL = 1
```

One blank line between imports and first declaration; two blank lines between
top-level declarations.

## Naming

- `snake_case` for functions, variables, parameters.
- `PascalCase` for structs, traits, type aliases.
- `SCREAMING_SNAKE_CASE` for `alias` constants.
- Leading underscore for module-private helpers (`_gemm_dispatch`).
- No `m_` / `p_` Hungarian prefixes.

## Struct definitions

```mojo
@value
struct TensorInfo:
    var ptr: UnsafePointer[Float32, MutExternalOrigin]
    var i8_ptr: UnsafePointer[Int8, MutExternalOrigin]
    var scale_ptr: UnsafePointer[Float32, MutExternalOrigin]
    var is_quantized: Bool
    var rows: Int
    var cols: Int
```

- Use `@value` for plain-data structs (auto copy/move/init).
- Field order: pointers first, then scalars, then flags. Groups matter for
  cache locality when the struct goes through the GPU packer.
- Document **shape invariants** on non-obvious fields via a short comment.

## Function signatures

```mojo
@always_inline
fn forward_sliding_attention[
    B: ComputeBackend, K: KVCacheTrait,
](
    mut backend: B,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    weights: LayerWeights,
    layer_idx: Int,
    pos: Int,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
):
```

- **Parametric functions** (over `B: ComputeBackend`, etc.) get compiled per
  backend. Zero dispatch overhead.
- Kernels take output pointer as first runtime arg (after parameters), then
  input pointers, then weights, then sizes.
- Use `mut backend: B` when the backend holds mutable state (scratch
  allocators, device handles).
- `@always_inline` on small hot helpers (`_gemm_dispatch`). Don't decorate
  everything — inliner noise.
- Return types: omit for void, explicit for everything else.

## Pointer hygiene

- `UnsafePointer[T, Origin]` — always specify origin.
  - `MutExternalOrigin` for Python-owned memory (via FFI).
  - `MutAnyOrigin` for pointers derived inside Mojo.
- Reconstruct FFI pointers once at the call boundary:
  ```mojo
  var ptr = UnsafePointer[Float32, MutExternalOrigin](
      unsafe_from_address=Int(py=llm["_some_key"])
  )
  ```
- Never reinterpret pointers mid-function — extract, use, done.
- Single-buffer arithmetic: scratch pointer math must stay within one
  `DeviceBuffer`. Multi-buffer moves require `enqueue_copy` which breaks the
  pattern.

## Backend dispatch

- Kernels are generic on `B: ComputeBackend`. Call via `backend.rms_norm(...)`,
  `backend.vec_mat_mul(...)` — never branch on `has_accelerator()` inside a
  kernel.
- `@comptime if has_accelerator():` gates compilation of GPU-only helpers at
  the module level. Nothing GPU-related leaks into CPU-only builds.
- `rebind[GPUContext](...)` casts a backend-erased parameter back to its
  concrete type inside a GPU branch. Use sparingly — it's a safety valve.

## Error handling

- Mojo doesn't have Python-style exceptions in kernels. Return codes or
  in-struct error flags where needed. Most kernels are infallible — caller
  ensures shapes.
- At FFI boundaries, failures propagate via the return value of `step_mojo`
  (non-zero = error).

## GPU specifics

- Weight staging in `gpu_context.mojo`. Layer kernels never upload — they
  consume what the packer put in the staging buffer.
- State swaps: use `backend.copy()` (kernel launch), not scalar loops.
- KV cache is a single arena across all layers. Index math lives in the
  cache trait implementation, not scattered in kernels.

## Comments

- Default to no comments. Code + good names.
- A one-line comment above a non-obvious block is fine when the WHY isn't
  evident from the code.
- No TODO/FIXME without a Beads issue ID (`# TODO(bd-123): swap to tiled matmul`).

## Testing

- Test files in `src/mo/tests/` named `test_*.mojo`.
- One `fn test_*()` per behavior.
- Use deterministic fixtures — no random init without a fixed seed.
- Test across backends by parameterizing the test helper.

## Anti-patterns (reject on review)

- GPU code without `@comptime if has_accelerator()` gating at the module
  level.
- Dereferencing FFI pointers without first checking alignment / contiguity in
  the caller.
- Branching on backend type inside a kernel (`if isinstance(backend, GPUBackend)`).
- Multiple `DeviceBuffer` allocations per layer on the streaming path.
- Stateful globals — pass state through the `llm` dict or function args.
- `@always_inline` on cold paths (bloats binary).
- Comments describing WHAT the code does.

## Patterns specific to mogemma

### FFI entry point

```mojo
fn step_mojo(llm: PythonObject, ...) -> Int:
    var gpu_initialized = Bool(llm["_gpu_initialized"])
    if gpu_initialized:
        return _run_step[GPUBackend, GPUKVCache, S, C, P](llm, ...)
    else:
        return _run_step[CPUBackend, CPUKVCache, 0, 0, 0](llm, ...)
```

CPU callers pass `0` for GPU parameter keys — zero overhead (dead
elimination at compile time).

### Weight staging

```mojo
@comptime if has_accelerator():
    upload_layer_weights(rebind[GPUContext](ctx), weights, layer_idx)
```

### Struct flattening for packers

Layer-weights structs implement a `_flatten` sibling function in `core.mojo`
that writes pointers in a stable order. Packers and hydrators must agree on
that order — if you reorder fields, update both halves together.
