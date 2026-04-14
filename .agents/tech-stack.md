# Tech Stack — mogemma

Authoritative list of tech choices. Changes here must precede
implementation changes. Cross-reference: `pyproject.toml` is the machine-
readable source of truth for pins; this doc explains the why.

## Core languages & runtimes

- **Mojo:** `>=0.26.1a1` (build-time pin in `pyproject.toml`; nightly like
  `0.26.3.0.dev` used for GPU work). Installed via the custom uv extra index
  declared under `[tool.uv.index]`. Primary language for performance-
  critical kernels and forward-pass code.
- **Python:** 3.10 target, supported 3.10–3.14, CI-tested 3.10–3.13. Primary
  user-facing interface.

## Python tooling & quality

- **uv:** Package management + virtual env + Mojo compiler distribution.
- **Ruff:** Linter + formatter (`[tool.ruff]`; line length 120, Google
  docstring convention, preview features on).
- **Mypy:** Strict type checking (`[tool.mypy]`).
- **Pyright:** Second type checker, expectations aligned (not strict
  superset) (`[tool.pyright]`).
- **Pytest:** Test runner for Python + Mojo bridge tests. `anyio` for async
  tests (covers trio too).

## Build & packaging

- **Makefile:** Canonical command surface — `make lint`, `make test`,
  `make check-all`, `make build`, etc. Never invoke underlying tools
  directly; keep local/CI parity via `make`.
- **hatchling + hatch-mojo (`>=0.1.8`):** Compiles
  `src/mo/mogemma/core.mojo` → `src/py/mogemma/_core.so`. `bundle-libs =
  true` is the canonical wheel policy.
- **cibuildwheel:** Multi-platform wheel builds. Linux via
  `manylinux_2_34` (AlmaLinux 9); macOS x86_64 + arm64; aarch64 via QEMU.
  See `.agents/knowledge/build-and-packaging.md` for libstdcxx quirks.
- **bump-my-version:** Release version string management.

## FFI bridge

- **Python ↔ Mojo:** numpy pointers passed across FFI as integers; Mojo
  reconstructs via `UnsafePointer[..., MutExternalOrigin]
  (unsafe_from_address=Int(py=...))`. Not zero-copy in the strict sense —
  Python retains the array; Mojo reads through the pointer; GC protection
  is the caller's responsibility.
- **Opaque handles:** GPU buffer pointers stored in a Python `llm` dict as
  `int(heap_ptr)`; never dereferenced on Python side.

## Storage, checkpoints, weights

- **obstore:** S3-compatible object storage client. `GCSStore` for
  `gs://gemma-data`, `HTTPStore` for Hugging Face, `LocalStore` for cache
  mirroring.
- **tensorstore + Orbax:** Read Gemma 4 OCDBT checkpoint format (as
  shipped by Google).
- **safetensors:** Runtime weight format after Orbax → safetensors
  conversion. mmap-backed, zero-copy read.
- **sentencepiece:** Tokenizer (via `[tool.hatch.envs.default.extras]` /
  optional-dep groups). Shared `tokenizer_gemma4.model` across all Gemma 4
  variants.

## Optional / extra dependencies

- **pillow:** Vision input preprocessing (`vision` extra).
- **librosa / numpy audio ops:** Mel spectrogram computation for audio
  inputs on E2B/E4B (`audio.py`).
- **opentelemetry-api:** Observability (`telemetry` extra).

## Models (primary target)

**Gemma 4** (active development target — enum `Gemma4Variant` in `model.py`):

| Variant | Enum | Params | Modalities | Notes |
|---|---|---|---|---|
| E2B | `DENSE_E2B` | ~2B | text + image + audio | Dense, PLE, double-wide MLP (4x) |
| E4B | `DENSE_E4B` | ~4B | text + image + audio | Dense, PLE, standard MLP (8x) |
| 26B-A4B-it | `MOE_26B_A4B` | 26B (4B active) | text + image | MoE two-branch (128 experts, top-8) + dense shared MLP |
| 31B | `DENSE_31B` | 31B | text + image | Dense |

All variants use hybrid attention (sliding-window + periodic full-attention
layers). See `.agents/knowledge/gemma4-models.md` for full architectural
detail.

**Gemma 3** (legacy): Retained compatibility for downstream users; not the
active development target.

## GPU

- Mojo stdlib `std.gpu.host` for device context + buffers.
- Backend abstraction: `trait ComputeBackend` with `CPUBackend` (always
  compiled) and `GPUBackend` (gated by `@comptime if has_accelerator()`).
- CUDA runtime required on target machine for GPU-enabled wheels.

## CI

- **GitHub Actions** (`.github/workflows/`):
  - `ci.yml`: lint + type-check + test matrix (Python 3.10–3.13).
  - `pr-title.yml`: conventional-commits enforcement.
  - `publish.yml`: PyPI publish via trusted publishing / OIDC.
  - `perf-benchmark.yml`: advisory benchmarks.

## Tech stack changes

When this file needs to change:

1. Discuss the change in the flow's PRD or in an RFC-style issue.
2. Update `tech-stack.md` **before** dependency changes land in
   `pyproject.toml`.
3. Refresh `.agents/knowledge/build-and-packaging.md` or related
   knowledge files if the change affects build or runtime.
4. Validate in a clean venv + clean CI before merging.
