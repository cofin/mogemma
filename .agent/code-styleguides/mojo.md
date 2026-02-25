# Mojo Style Guide (Universal Standards)

Guidelines for high-performance Mojo development, focusing on memory safety, SIMD, and Python interoperability.

## Core Principles

- **Prefer `fn` over `def`**: Use `fn` for strict type checking and performance. Only use `def` for dynamic Python-like behavior if necessary.
- **Strict Typing**: Always provide explicit types for function arguments and return values.
- **Memory Safety**: Leverage Mojo's ownership system (`owned`, `borrowed`, `inout`). Use `UnsafePointer` only when necessary for FFI or low-level optimizations.

## Project Structure (Hybrid Mojo-Python)

The standardized directory structure for hybrid libraries:
```text
project_root/
├── pyproject.toml     # Managed by uv (mandatory)
├── tools/
│   └── hatch_build.py # Automated Mojo compilation hook
├── src/
│   ├── mo/            # High-performance Mojo source code
│   │   └── package_name/
│   │       ├── __init__.mojo
│   │       ├── core.mojo
│   │       └── utils.mojo
│   └── py/            # Python wrapper & user-facing API
│       └── package_name/
│           ├── __init__.py
│           └── _core.so (Compiled Mojo extension)
```

## Function Definitions

```mojo
# Standard: fn with explicit types and docstrings
fn calculate_metrics(a: Float32, b: Float32) -> Float32:
    """Computes the metric for the given values.

    Args:
        a: First input value.
        b: Second input value.

    Returns:
        The computed Float32 result.
    """
    return a * b

# Error propagation
fn load_data(path: String) raises -> String:
    if not path:
        raise Error("Path cannot be empty")
    return "Data loaded"
```

## Performance & Vectorization (Learnings)

### SIMD (Single Instruction, Multiple Data)

Always use SIMD for numeric operations on arrays.

```mojo
from algorithm import vectorize
from sys.info import simdwidthof

alias type = DType.float32
alias width = simdwidthof[type]()

fn fast_process(ptr: UnsafePointer[Float32], size: Int):
    @parameter
    fn compute[w: Int](i: Int):
        var data = ptr.load[width=w](i)
        ptr.store[width=w](i, data * 2.0)

    vectorize[width, compute](size)
```

### Parallelization (GIL Bypassing)

Mojo has no GIL. Use `parallelize` for true multi-core scaling from Python.

```mojo
from algorithm import parallelize

fn heavy_compute(i: Int):
    # This executes in parallel without GIL overhead
    pass

fn run_parallel(tasks: Int):
    parallelize[heavy_compute](tasks)
```

### Zero-Copy Memory Sharing

Use `UnsafePointer` to share memory with NumPy without copying.

```mojo
from python import Python

fn process_numpy_array(array: PythonObject) raises:
    var np = Python.import_module("numpy")
    var np_array = np.asarray(array, dtype=np.float32)
    
    # Get direct pointer to data
    var data_ptr = UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(py=np_array.__array_interface__["data"][0])
    )
    
    # Process data_ptr directly...
```

## Python Extensions

### Creating the Bridge

Use `@export` and `PythonModuleBuilder` to expose Mojo functions to Python.

```mojo
from python.bindings import PythonModuleBuilder

@export
fn PyInit__core() -> PythonObject:
    try:
        var b = PythonModuleBuilder("_core")
        b.def_function[mojo_kernel]("kernel")
        return b.finalize()
    except e:
        # Proper error handling for initialization
        return None
```

## Build Automation (Learning)

The `tools/hatch_build.py` pattern is the standard for automated Mojo compilation during `uv build`.

```python
import subprocess
from pathlib import Path
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        if self.target_name != "wheel": return
        # Logic to find mojo and run 'mojo build --dylib'
```

## Anti-Patterns

- **Avoid `def` in compute loops**: Use `fn` for static performance.
- **Avoid manual memory copies**: Use `UnsafePointer` and NumPy's buffer protocol.
- **Avoid hardcoded paths**: Use `tools/` scripts to locate the Mojo compiler.
