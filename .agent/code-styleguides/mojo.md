# Mojo Style Guide

Guidelines for high-performance Mojo development, focusing on memory safety, SIMD, and Python interoperability.

## Core Principles

- **Prefer `fn` over `def`**: Use `fn` for strict type checking and performance. Only use `def` for dynamic Python-like behavior if necessary.
- **Strict Typing**: Always provide explicit types for function arguments and return values.
- **Memory Safety**: Leverage Mojo's ownership system (`owned`, `borrowed`, `inout`). Use `UnsafePointer` only when necessary for FFI or low-level optimizations.

## Project Structure

```text
my_project/
├── pyproject.toml     # Managed by uv
├── src/
│   ├── mo/            # Mojo source code
│   │   └── my_lib/
│   │       ├── __init__.mojo
│   │       ├── core.mojo
│   │       └── utils.mojo
│   └── py/            # Python wrapper
│       └── my_lib/
│           ├── __init__.py
│           └── _core.so (compiled)
```

## Function Definitions

```mojo
# Preferred: fn with explicit types and traits
fn calculate_sum(a: Int, b: Int) -> Int:
    """Adds two integers."""
    return a + b

# Handling errors
fn risky_op() raises -> String:
    if True:
        raise Error("Something went wrong")
    return "Success"
```

## Structs and Traits

Mojo structs are static and performant.

```mojo
@value
struct Point(CollectionElement):
    var x: Float32
    var y: Float32

    fn __init__(inout self, x: Float32, y: Float32):
        self.x = x
        self.y = y

# Trait definition
trait Shape:
    fn area(self) -> Float32: ...
```

## Performance & Vectorization

### SIMD (Single Instruction, Multiple Data)

Always use SIMD for numeric operations on arrays.

```mojo
from algorithm import vectorize
from sys.info import simdwidthof

alias type = DType.float32
alias width = simdwidthof[type]()

fn fast_add(ptr_a: UnsafePointer[Float32], ptr_b: UnsafePointer[Float32], size: Int):
    @parameter
    fn vectorize_step[w: Int](i: Int):
        var a = ptr_a.load[width=w](i)
        var b = ptr_b.load[width=w](i)
        ptr_a.store[width=w](i, a + b)

    vectorize[width, vectorize_step](size)
```

### Parallelization

Mojo has no GIL. Use `parallelize` for multi-core scaling.

```mojo
from algorithm import parallelize

fn heavy_computation(i: Int):
    # Do work for index i
    pass

fn run_parallel(total_tasks: Int):
    parallelize[heavy_computation](total_tasks)
```

## Python Interoperability

### Calling Python from Mojo

```mojo
from python import Python

fn use_numpy() raises:
    var np = Python.import_module("numpy")
    var array = np.array([1, 2, 3])
    print(array.mean())
```

### Creating Python Extensions

Use `@export` and `PythonModuleBuilder` to expose Mojo functions to Python.

```mojo
from python.bindings import PythonModuleBuilder

@export
fn PyInit_my_extension() -> PythonObject:
    try:
        var b = PythonModuleBuilder("my_extension")
        b.def_function[my_mojo_fn]("my_mojo_fn")
        return b.finalize()
    except e:
        # Handle initialization error
        pass
```

## Memory Management

- Use `@value` for simple data containers to get auto-generated lifecycle methods.
- Use `owned` for passing ownership (transferring memory).
- Use `borrowed` (default for `fn`) for read-only access.
- Use `inout` for mutable references.

## Import Organization

1.  Mojo Standard Library (`from sys import ...`)
2.  Python Interop (`from python import ...`)
3.  Local Mojo Modules (`from .utils import ...`)

## Anti-Patterns

- **Avoid `def` in hot paths**: It introduces overhead similar to Python.
- **Avoid manual pointer arithmetic** where slices or higher-level abstractions suffice.
- **Don't ignore `raises`**: Always handle or propagate errors.
