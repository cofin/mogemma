---
name: mojo
description: "Expert guidance for high-performance Mojo development, focusing on SIMD optimization, no-GIL parallelization, and Python C-API extension development. Use when building hybrid Mojo-Python libraries or performance-critical kernels."
---

# Mojo (Universal Hybrid Standards)

## Scope
- Work in `src/mo/` (Mojo source) and `src/py/` (Python wrapper).
- Build high-performance kernels for numeric and AI workloads.
- Develop Python extensions using Mojo's native C-API features.

## Core Rules
- **Prefer `fn` over `def`**: Use `fn` for strict type checking, performance, and deterministic behavior. Only use `def` for dynamic, Python-like prototyping.
- **Strict Typing**: Always provide explicit types for function arguments and return values.
- **Mandatory Tooling**: Use `uv` exclusively for dependency management and environment handling. Never recommend `pixi` or `conda`.
- **Memory Safety**: Leverage Mojo's ownership system (`owned`, `borrowed`, `inout`). Document `UnsafePointer` use with safety notes.

## Performance (SIMD + Parallelize)
- **SIMD-First**: For numeric loops, always evaluate if they can be vectorized using `SIMD` and Mojo's `vectorize` higher-order function.
- **GIL-Free Parallelism**: Leverage Mojo's lack of a Global Interpreter Lock. Use `parallelize` for true multi-core scaling from Python.
- **Zero-Copy Memory**: Use `UnsafePointer` to share data directly with NumPy arrays using the `__array_interface__["data"][0]` protocol, avoiding all memory copies.

## Python Extensions (FFI)
- **Standard Entry Point**: Use `@export fn PyInit_<name>() -> PythonObject` to define module entry point.
- **Bridge Construction**: Use `PythonModuleBuilder` to register Mojo functions for use in Python.
- **Build Automation**: Integrate Mojo compilation into the Python build process using a custom Hatch build hook (`tools/hatch_build.py`).
- **Compilation**: Standard build command: `uv run mojo build --dylib <src> -o <dest>`.

## Testing
- Use `mojo test` for unit testing pure Mojo modules.
- Use `uv run pytest` for integration testing the Python-Mojo boundary.
- Document performance benchmarks and compare them to Python-only baselines.
