# Product Guidelines — mogemma

Principles for how we build, document, and present mogemma. Keep short;
full detail lives in `.agents/knowledge/` and `.agents/code-styleguides/`.

## Development philosophy

- **Dead simple public API.** Complexity hides in the runtime, not the
  surface. New surface area requires a PRD entry and an update to
  `.agents/knowledge/python-runtime.md`.
- **Explicit over implicit.** Defaults should work out of the box, but
  memory layout, device selection, variant resolution, and error paths
  stay visible. No silent fallbacks.
- **Concise, action-oriented docs.** Code-first. Minimize prose. README,
  docstrings, and examples should favor runnable snippets over
  exposition.

## Project structure

- **Python package:** `src/py/mogemma/` — user-facing API + orchestration.
- **Mojo package:** `src/mo/mogemma/` — kernels, forward-pass, GPU context.
- **Python tests:** `src/py/tests/`.
- **Mojo tests:** `src/mo/tests/`.
- **Build artifact:** `src/py/mogemma/_core.so` (produced by `make build`).

Packaging is hatchling + hatch-mojo; orchestration via `Makefile` (see
`.agents/workflow.md` for canonical commands).

## Technical standards

- **Quality gates** (see `.agents/knowledge/quality-gates.md`):
  - **Typing:** strict `mypy` + `pyright`; no `Any` in public API without
    justification.
  - **Linting:** `ruff` (preview on, line length 120, Google docstrings).
  - **Mojo:** built-in formatter, line length 120.
  - **Testing:** pytest, Google-style; deterministic fixtures for parity
    tests.
  - **Zero warnings:** hard-fail on any gate.
- **FFI discipline** (see `.agents/knowledge/mojo-runtime.md` and
  `patterns.md`):
  - Rectangular arrays only across FFI; pad in Python before handing
    pointers to Mojo.
  - Retain numpy references for the lifetime of any Mojo-held pointer.
  - Specify `UnsafePointer` origins explicitly.
- **Error surface:** explicit error taxonomy (`ModelNotFoundError`,
  tokenizer-missing, core-missing, runtime). Distinguish Python-layer vs
  Mojo-layer failures in messages.

## Performance & memory

- **Streaming weight pipelines.** Per-layer weights flow through a
  reusable staging buffer — no per-call allocations on the hot path.
- **Single KV-cache arena** across all layers; no per-layer allocation
  sprawl.
- **GPU gating.** `@comptime if has_accelerator()` keeps CPU-only builds
  free of GPU symbols.
- **Validation at FFI boundary.** Shape checks, dtype checks, contiguity
  checks in Python before calling into Mojo. Segfaults from the Mojo
  side are treated as bugs, not "user error."

## Scope discipline

- Tasks that grow beyond the original plan get refined in the spec, not
  silently descoped. `NotImplementedError` stubs that pass tests are
  rejected on review.
- Changes to public API require matching updates to
  `.agents/knowledge/python-runtime.md` in the same PR.
- Changes to Mojo struct layouts require matching updates to
  `.agents/knowledge/mojo-runtime.md` and the GPU packer in
  `gpu_context.mojo` — all in one PR.

## Related

- `.agents/workflow.md` — canonical process and commands
- `.agents/patterns.md` — elevated actionable rules
- `.agents/tech-stack.md` — approved technologies
- `.agents/knowledge/README.md` — component doc index
- `.agents/code-styleguides/python.md`, `mojo.md`, `testing.md`, `bash.md`
