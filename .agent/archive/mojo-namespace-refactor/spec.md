# Flow: mojo-namespace-refactor

## Specification

### Goal
Restructure the Mojo source files by moving them into a `mogemma` package namespace inside `src/mo/`, to mirror the Python directory layout. Ensure all build tools, tests, and internal imports are correctly updated to reflect the new structure.

### Code Analysis Summary
- The primary Mojo files currently reside at the root of `src/mo/`: `core.mojo`, `model.mojo`, `layers.mojo`, `ops.mojo`.
- These files reference each other via flat module imports (e.g., `from model import ModelWeights`).
- `tools/hatch_build.py` compiles `src/mo/core.mojo` using `-I src/mo` to produce `_core.so`.
- `Makefile` has an explicit target `build-mojo` that also compiles `src/mo/core.mojo` using `-I src/mo` (implicitly or explicitly).
- Tests are located in `src/mo/tests/` and import the flat modules directly.

### Requirements
- Move `core.mojo`, `layers.mojo`, `model.mojo`, and `ops.mojo` to `src/mo/mogemma/`.
- Add an empty (or minimally declarative) `__init__.mojo` to `src/mo/mogemma/` to formalize the package.
- Update internal imports in the moved files to use the `mogemma` namespace (e.g., `from mogemma.model import ...`).
- Update Mojo tests in `src/mo/tests/` to use `mogemma` imports.
- Update `tools/hatch_build.py` to point to `src/mo/mogemma/core.mojo` as the build target.
- Update `Makefile` to target `src/mo/mogemma/core.mojo`.

### Non-Goals
- No changes to underlying logic, mathematical models, or transformer architecture.
- No changes to the location or internal behavior of the Python code (`src/py/mogemma/`).

## Implementation Plan

### Phase 1: Directory Structure and File Movement
- [x] 1.1 Create `src/mo/mogemma/` directory.
- [x] 1.2 Move `core.mojo`, `layers.mojo`, `model.mojo`, and `ops.mojo` into `src/mo/mogemma/`.
- [x] 1.3 Create an empty `src/mo/mogemma/__init__.mojo` file.

### Phase 2: Update Internal Imports
- [x] 2.1 Update `src/mo/mogemma/core.mojo` imports (e.g., `from mogemma.model import ...`, `from mogemma.layers import ...`).
- [x] 2.2 Update `src/mo/mogemma/layers.mojo` imports (e.g., `from mogemma.model import ...`, `from mogemma.ops import ...`).
- [x] 2.3 Update `src/mo/mogemma/model.mojo` and `ops.mojo` if they have any internal cross-imports.

### Phase 3: Update Tests
- [x] 3.1 In `src/mo/tests/test_model.mojo`, change `from model` to `from mogemma.model`.
- [x] 3.2 In `src/mo/tests/test_layers.mojo`, change `from layers` to `from mogemma.layers` (and model/ops as needed).
- [x] 3.3 In `src/mo/tests/test_ops.mojo`, change `from ops` to `from mogemma.ops`.

### Phase 4: Build Tooling Updates
- [x] 4.1 Update `tools/hatch_build.py` line 20: `mojo_src = root / "src" / "mo" / "mogemma" / "core.mojo"`.
- [x] 4.2 Update `Makefile` line 49: `uv run mojo build --emit shared-lib src/mo/mogemma/core.mojo -o src/py/mogemma/_core.so`.
- [x] 4.3 Verify `make build-mojo` succeeds locally.
