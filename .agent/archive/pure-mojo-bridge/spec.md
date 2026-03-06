# Flow: pure-mojo-bridge

## Specification

### Goal
Strip out all dependencies on the Modular `max` Python framework and establish a direct zero-copy memory bridge between Python's `safetensors` weight loader and Mojo's native `Tensor` objects.

### Code Analysis Summary
1. `pyproject.toml` relies heavily on `modular` and `max` for the backend, pulling in a massive Python footprint.
2. `src/py/mogemma/model.py` currently delegates initialization to `_core.init_model(model_path)`, which uses `max` internals.
3. `src/mo/core.mojo` uses Python `max` entrypoints like `max.entrypoints.llm`.
4. We need to implement a Python-side weight loader using `safetensors` to `mmap` the Gemma 3 weights, and extract memory pointers.
5. Mojo needs an FFI update to accept these raw integer pointers and wrap them into native Mojo structures.

### Requirements
1. Remove `modular` and `max` dependencies and indexes from `pyproject.toml`.
2. Add `safetensors` to `pyproject.toml` base dependencies.
3. Implement `src/py/mogemma/loader.py` to `mmap` load `.safetensors` files and expose memory addresses and shapes.
4. Update `src/mo/core.mojo` to expose a new `init_model` that accepts these pointers/shapes instead of a model path.
5. Guarantee zero-copy: weights must not be duplicated into Mojo memory, but mapped directly using Mojo pointers.

### Non-goals
- Implementing transformer math (that is Chapter 2 & 3).
- Implementing KV cache (Chapter 4).
- Actually executing the model (just bridging the weight memory).

## Implementation Plan

### Phase 1: Dependency Purge
- [x] 1.1 Remove `modular` from `pyproject.toml` `optional-dependencies` and `sources`.
- [x] 1.2 Remove the `modular-nightly` index from `pyproject.toml`.
- [x] 1.3 Add `safetensors` to `pyproject.toml` base `dependencies`.
- [x] 1.4 Update lockfile (`uv lock`) to reflect the purged dependencies.

### Phase 2: Python Weight Loader
- [x] 2.1 Create `src/py/mogemma/loader.py`.
- [x] 2.2 Implement a `SafetensorsLoader` class that opens `.safetensors` files using `safetensors.safe_open`.
- [x] 2.3 Add logic to extract tensor names, shapes, dtypes, and raw memory pointers (`tensor.data_ptr()`).
- [x] 2.4 Handle multi-file safetensors (e.g., parsing `model.safetensors.index.json` if necessary).

### Phase 3: Mojo Zero-Copy FFI Bridge
- [x] 3.1 Update `src/mo/core.mojo` to remove all `max.entrypoints` references.
- [x] 3.2 Define a Mojo `Struct` or function that accepts a Python dictionary of tensor metadata (name -> [pointer, shape]).
- [x] 3.3 Use Mojo's `DTypePointer` or `UnsafePointer` to cast the raw integer pointers passed from Python.
- [x] 3.4 Create a basic test function in Mojo that reads a few bytes from the bridged pointers to verify memory alignment and zero-copy success.
