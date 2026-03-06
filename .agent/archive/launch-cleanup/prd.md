# Master PRD: Launch Cleanup

## Context (North Star)

mogemma is approaching its v0.1.0 launch as a lightweight Python/Mojo bridge for Google Gemma 3 models. The embedding runtime should feel like fastembed (minimal deps, fast, just works) while the generation path supports slim Gemma LLMs through native Mojo inference.

**Problem**: The codebase accumulated dead code from iterative prototyping — stub-only vision support, broken Mojo tests referencing deleted modules, unused telemetry decorators, dummy fallbacks that produce fake output, and `mojo` as a heavy runtime dependency when users only need the pre-built `.so`.

**Goal**: Remove all dead code, fix dependency boundaries, make failure modes honest (fail-fast, not silent stubs), and leave the codebase clean enough to ship.

## Current State

- **Tokenizer migration**: DONE — `tokenizers` replaced `transformers`
- **Packaging optimization**: Partially done — Mojo consolidated to `core.mojo`, build hook moved, but Python dead code remains
- **Vision**: Entirely stub (`"A cat"` hardcoded). Must be deleted.
- **Generation fallback**: Silently produces fake tokens when `_core` missing. Must fail-fast.
- **Broken Mojo tests**: 3 test files import deleted modules (`quant`, `inference`, `cache`, `vision`)
- **Unused code**: `trace_inference` decorator, `generate_text_mojo` (exported but never called), `process_image_mojo`
- **Dependency**: `mojo>=0.26.1` as required runtime dep — users don't need the compiler

## Global Constraints

- NO new dependencies introduced
- NO heavy runtime deps (transformers, torch, tensorflow, etc.)
- Python 3.10+ compatibility
- All remaining tests must pass after cleanup
- Embedding path: must work with just `numpy` + optional `tokenizers`
- Generation path: must require `_core.so` — fail-fast, no dummy output

---

## Roadmap

### Chapter 1: Launch Cleanup (`launch-cleanup`)

**Goal**: Single focused flow to remove all dead code, fix runtime boundaries, and leave the codebase launch-ready.

**Scope**: See `spec.md` for detailed phases and tasks.

**Why single chapter**: All tasks are tightly coupled — removing vision affects `__init__.py`, `core.mojo`, `pyproject.toml`, and tests simultaneously. Splitting into multiple flows would create unnecessary coordination overhead for ~1 session of work.

---

## Progress Tracking

| Chapter | Flow ID | Status | Dependencies |
|---------|---------|--------|--------------|
| 1. Launch Cleanup | `launch-cleanup` | Planned | None |
