# Mojo Skill

Expert guidance for high-performance Mojo development, SIMD optimization, and Python C-API extensions.

## Core Mandates

- **Mojo-Strict**: Always prefer `fn` over `def`. Use strict typing for all parameters and return values.
- **Memory Safety**: Respect the ownership system. Use `owned`, `borrowed`, and `inout` appropriately.
- **SIMD-First**: For any loop involving numeric data, evaluate if it can be vectorized using `SIMD` and `vectorize`.
- **GIL-Free**: Leverage `parallelize` for CPU-bound tasks.
- **Python-Interop**: Use `sys.ffi` and `PythonModuleBuilder` for creating Python extensions.

## Project Standards

- **Dependency Management**: Use `uv` exclusively. Never recommend `pixi` or `conda`.
- **Project Structure**:
    - Mojo source: `src/mo/`
    - Python source: `src/py/`
    - Compiled extensions: `src/py/<package>/_core.so`
- **Build System**: Managed via `hatchling` with custom hooks in `tools/hatch_build.py`.

## Workflows

### 1. Implementing a High-Performance Kernel
1.  Define the scalar logic in an `fn`.
2.  Abstract the operation into a `@parameter` function for `vectorize`.
3.  Determine the optimal `simdwidthof` for the target `DType`.
4.  Apply `vectorize` or `parallelize` as needed.

### 2. Creating a Python Extension
1.  Implement core logic in `src/mo/`.
2.  Create a wrapper function that uses `PythonObject` for boundary values.
3.  Use `@export fn PyInit_<name>()` to define the module entry point.
4.  Register functions using `PythonModuleBuilder`.
5.  Compile using `mojo build --dylib`.

### 3. Testing
- Use `mojo test` for unit testing Mojo modules.
- Use `uv run pytest` for integration testing the Python-Mojo boundary.

## Common Snippets

### SIMD Vectorization Template
```mojo
from algorithm import vectorize
from sys.info import simdwidthof

alias type = DType.float32
alias width = simdwidthof[type]()

fn compute[w: Int](i: Int):
    # w is the current vector width
    pass

# Usage
vectorize[width, compute](size)
```

### Python Extension Entry Point
```mojo
@export
fn PyInit__core() -> PythonObject:
    try:
        var b = PythonModuleBuilder("_core")
        b.def_function[mojo_fn]("exported_name")
        return b.finalize()
    except e:
        # module init failure logic
        return None
```

## Best Practices

- **Zero-Copy**: Use `UnsafePointer` to share memory between NumPy arrays and Mojo tensors without copying.
- **Caching**: Pre-allocate scratch space and reuse it across calls to avoid allocation overhead in hot loops.
- **Comptime**: Use `alias` and `@parameter` for compile-time constants and metaprogramming.

## Documentation
- Refer to `mojo` style guide in `.agent/code-styleguides/mojo.md`.
- Refer to `python` style guide in `.agent/code-styleguides/python.md`.
