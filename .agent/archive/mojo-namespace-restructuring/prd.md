# Master PRD: Mojo Namespace Restructuring

## Context
The Mojo source files for `mogemma` currently reside directly in `src/mo/` as freestanding files (`core.mojo`, `model.mojo`, `layers.mojo`, `ops.mojo`). This flat structure differs from the Python layout (`src/py/mogemma/`) and prevents the files from behaving as a well-structured package namespace. Moving them into a proper `mogemma` Mojo module will keep the code organized, clarify import boundaries, and ensure consistency between Python and Mojo paradigms.

## Roadmap

### Chapter 1: `mojo-namespace-refactor`
Move `core.mojo`, `model.mojo`, `layers.mojo`, and `ops.mojo` into a new `src/mo/mogemma/` directory. Add an `__init__.mojo` file to formalize the module. Update all internal imports in these files to use the new `mogemma` package structure. Fix build and test runners (`tools/hatch_build.py`, `Makefile`) to target the relocated entry point while maintaining the existing `src/mo/tests/` location.

## Global Constraints
1. **No Logic Changes:** This is purely a structural refactoring. No underlying mathematical or architectural logic should be modified.
2. **Build Parity:** The `hatch_build.py` build hook must continue emitting the `_core.so` shared library correctly.
3. **Test Stability:** Mojo tests currently in `src/mo/tests/` must run and pass without issue after updating their imports.
