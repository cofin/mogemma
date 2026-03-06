# Tech Stack - mogemma

## Core Languages & Runtimes
- **Mojo (v26.1+):** Primary language for performance-critical kernels and model orchestration.
- **Python (3.10+):** Primary interface for users, following the `litecore` baseline.
- **MAX Framework (Modular):** Native backend for Gemma 3 model execution.

## Python Tooling & Quality
- **uv:** For Python package management and virtual environments.
- **Ruff:** Linter and formatter (mirroring `litecore` config).
- **Mypy & Pyright:** Strict static type checking.
- **Pytest:** Test runner for the Python/Mojo bridge.

## Build & Orchestration
- **Makefile:** Standard orchestration for building, testing, and linting.
- **Python-to-Mojo Interop API:** Native bridge for zero-copy data exchange.

## Models
- **Google Gemma 3:** Native support for 4B, 12B, and 27B variants (Text, Vision, Embeddings).
