# Product Guidelines - mogemma

## Development Philosophy
- **Prose Style:** Concise, action-oriented, and focused on quick-start examples. Documentation should minimize fluff and prioritize code snippets.
- **Pythonic DX:** A "Dead Simple" high-level API that feels like established libraries (`transformers`, `sentence-transformers`) but with Mojo-powered performance.
- **Explicit over Implicit:** While defaults should work "out of the box," critical paths (memory usage, model paths) should be clear to avoid surprises.

## Project Structure (Mirroring `litecore`)
- **Python:** Located in `src/py/mogemma`.
- **Mojo:** Located in `src/mo/`.
- **Tooling:** Managed by `uv`, using `Makefile` for orchestration.

## Technical Standards
- **High-Performance Bridge:** Prioritize **Zero-Copy Buffers** using Mojo's latest Python-to-Mojo interoperability API to ensure near-zero overhead.
- **Strict Quality Gates:** 
  - **Typing:** `mypy` and `pyright` in strict mode for all Python code.
  - **Linting:** `ruff` for linting and formatting, using the configuration patterns from the `litecore` project (Line length 120, Google docstring convention, etc.).
  - **Testing:** `pytest` for comprehensive unit and integration tests.
- **Error Handling:** Informative, human-readable error messages with suggested fixes. Explicitly distinguish between Mojo-level and Python-level errors.

## Performance & Memory
- **Efficiency:** Leverage Mojo's explicit memory management and "eager-style" Tensor APIs for intuitive but fast data handling.
- **Validation:** Strict validation at the Mojo/Python boundary to ensure safety and prevent segfaults.
