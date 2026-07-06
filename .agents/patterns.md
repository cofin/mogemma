# Project Patterns

> Consolidated learnings and patterns from all flows.
> This file is the single source of truth for project conventions.

## Code Conventions

- **Mojo Style:** Use `snake_case` for variables and functions.
- **Python Typing:** Always use `>=3.10` types (built-in types like `list`, `dict`).
- **Docstrings:** Required for all public functions/methods.

## Architecture Patterns

- **Hybrid Mojo-Python:** Use `hatch-mojo` plugin (configured in `pyproject.toml` under `[[tool.hatch.build.targets.wheel.hooks.mojo.jobs]]`) for Mojo compilation integration.
- **Contract Testing:** Use `test_altup_contract.mojo` style for testing cross-language boundaries.
- **Packaging Baseline:** Use `hatch-mojo>=0.1.8` with `bundle-libs = true` as canonical wheel policy; only keep Linux-specific overrides when they have explicit rationale and exit criteria.
- **GPU Backend Design:** 
  - Use `comptime if has_accelerator()` to gate GPU code.
  - Standard Mojo GPU API: `std.gpu.host` for context and buffers.
  - Trait-based backends: `trait ComputeBackend` with `@staticmethod def` for CPU, and explicit `GPUBackend` for kernel launches requiring `DeviceContext`.
  - Contiguous Memory Allocation: Use single contiguous arena + per-layer offsets for KV caches to avoid allocation sprawl.
- **OrbaxLoader Stub:** Use `OrbaxLoader.__new__(cls)` stub pattern (populating `.model_path` only) to expose `_enumerate_tensor_names` and `_open_tensor` without triggering eager loads. (from: orbax-safetensors-conversion)

## Gemma Model Patterns

- **Instruction-Tuned (`*-it`) Models:** Require turn formatting (`<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n`) for reliable output. Raw prompts produce low-quality artifacts.
- **EOS Handling:** Prioritize `<end_of_turn>` as EOS for `*-it` models; otherwise generation drifts into repeated turn markers.
- **Deterministic Validation:** Use `temperature=0, top_k=1` for factual contract checks; stochastic decoding masks regressions.
- **Multimodal FFI:** Use `step_with_embedding` abstraction to inject pre-computed vision/audio vectors directly into the decoder.
- **Hybrid KV Cache:** Gemma 4 uses hybrid sliding-window + global attention; ring buffer implementation for windowed entries is valid without special causal masking after initial wrap.
- **PLE Norm Layout:** Post-layer PLE norm is `layer_N.post_per_layer_input_norm.scale` (shape `(1536,)`), not the global `embedder.per_layer_projection_norm.scale` (shape `(256,)`). (from: orbax-safetensors-conversion)

## GPU Forward Path Patterns

- **Weight streaming separation**: Step/encoder functions orchestrate weight upload (`comptime if has_accelerator()` → `rebind` → `upload_*_weights` → `sync`); layer functions are backend-agnostic and receive device pointers. (from: gpu-forward-paths)
- **Scalar loops → backend.copy()**: Replace scalar state-swap loops (`for i in range(n): dst.store(i, src.load(i))`) with `backend.copy(dst, src, n)` for GPU compatibility. (from: gpu-forward-paths)
- **GPU KV cache reset requires kernel**: Can't zero device memory from host; `GPUKVCache.reset(mut ctx)` launches a zero-fill kernel. (from: gpu-integration-dispatch)
- **Inline GPU handle retrieval**: GPU pointer retrieval from `llm` dict uses inline `UnsafePointer[T](unsafe_from_address=Int(py=llm["_key"]))` — extracting into helpers adds `comptime` guard complexity without benefit. (from: gpu-integration-dispatch)
- **Shared VisionLayerWeights for audio**: Both vision and audio encoders reuse `upload_vision_layer_weights` and `VisionLayerWeights` struct — audio layers have the same weight shape. (from: gpu-forward-paths)
- **Kernel launch for intra-buffer copies**: `copy_state` uses a kernel launch (not `enqueue_copy`) because scratch buffer uses pointer arithmetic within a single `DeviceBuffer`, not separate handles. (from: gpu-forward-paths)
- **No `_step_gpu` extraction needed**: `_run_step[B, K, S, C, P]` already abstracts GPU/CPU dispatch via type parameters — a separate `_step_gpu` wrapper adds nothing. (from: gpu-integration-dispatch)

## Gotchas & Warnings

- **Mojo Pathing:** Ensure `MOJO_PATH` includes `src/mo`.
- **Hatch/UV:** Use `uv run` to ensure virtualenv consistency.
- **Model Rebuild Overhead:** Cache runtime descriptors at init; hot-path model rebuild per-call degrades performance.
- **FFI Rectangular Arrays:** FFI to Mojo from Python requires purely rectangular arrays when passing multi-dimensional arrays (like token sequences). The Python layer must proactively check for heterogeneous sequence lengths and enforce padding before delegating to the backend.
- **Mojo 24.x/26.x+ Modernization:** 
  - Use `comptime` instead of `alias` for constants.
  - Use `def` for all functions (including traits and @staticmethod).
  - Use `[]` for list initialization.
  - Tuple returns `(T1, T2)` are deprecated in nightly; refactor to structs.
  - **UnsafePointer Origins:** Always specify origins for `UnsafePointer` (e.g., `MutExternalOrigin` for heap, `MutAnyOrigin` for widened pointers).
  - **Type Erasure:** Use `rebind[TargetType](value)` to cast between compatible types, especially when passing GPU handles between Python and Mojo.
  - **List Transfer:** Use `^` (transfer operator) when returning a `List` from a function to avoid implicit copy errors.
  - **HostBuffer Pointer Retrieval:** In modern Mojo, `HostBuffer.unsafe_ptr()` returns an `UnsafePointer` directly. Legacy `.value()` accessors do not compile; perform pointer arithmetic, `load`, and `store` directly on the returned pointer. (from: orbax-safetensors-conversion)
  - **Address-backed nullable pointers:** Do not construct null `UnsafePointer` values. Store nullable tensor/device pointers as integer addresses on structs and reconstruct `UnsafePointer` values only at nonzero use sites. (from: gemma4-working-order-recovery)
- **FFI Memory Safety:** Keep references to converted numpy arrays in Python to prevent GC before Mojo consumption.
- **HuggingFace Shard Discovery:** HF doesn't support bucket listing; parse `model.safetensors.index.json` instead.
- **HuggingFace URL Normalization:** When using `obstore.store.HTTPStore`, ensure the base URL does *not* have a trailing slash (e.g., `.../resolve/main`) to avoid double-slash 404 errors (`.../resolve/main//file`) from Hugging Face.
- **Obstore download streaming:** `obstore.get()` and `obstore.get_async()` both return `GetResult`; call `result.stream()` and iterate the returned `BytesStream` rather than materializing `bytes_async()` for large downloads. (from: gemma4-working-order-recovery)
- **Gemma 4 Default Selection:** `GenerationConfig` / `EmbeddingConfig` default to `google/gemma-4-E4B-it` — the latest small text/image PLE variant available through the public GCS catalog. Audio is recognized but not implemented in this runtime.
- **Gemma 4 Vision Cropping:** Target budgets must be cropped to largest multiple of `patch_size` (16) after resizing (e.g., 280 -> 272).
- **Beads FK Constraints:** Large Beads DBs may hit foreign key corruption; favor markdown-only tracking for rapid feature development if instability occurs.
- **Orbax GCS config.json Missing:** GCS does not ship `config.json` for E2B-it checkpoints; the conversion process must synthesize it. (from: orbax-safetensors-conversion)
- **E2B-it Disk Space Budget:** E2B-it checkpoints are ~17GB; budget disk space carefully when managing multi-variant local workflows. (from: orbax-safetensors-conversion)
- **Contract Resolution via Consumer:** When cross-language FFI contracts are ambiguous from loader code, read the forward-pass consumer tensor shapes (e.g. `vec_mat_mul` and `rms_norm` parameters) to determine the layout. (from: orbax-safetensors-conversion)

## Testing Patterns

- **Pytest-Mojo:** Use `pytest` to drive Mojo tests via the standard CLI.
- **Contract Tests:** Cross-language boundaries tested via `test_altup_contract.mojo` style.
- **Python test layout:** Keep Python tests under `src/py/tests/unit/` for pure unit/API behavior and
  `src/py/tests/integration/` for filesystem-heavy, native-extension, live-network, workflow, or real-checkpoint paths.
- **Mojo test layout:** Keep Mojo tests under `src/mo/tests/unit/` for pure CPU/runtime/kernel-shape behavior and
  `src/mo/tests/integration/` for Python interop, GPU-gated, native-boundary, or hardware-sensitive paths. The
  `src/mo/tests/test_mojo.py` pytest harness discovers `.mojo` modules recursively.
- **No absence-only scar tests:** Do not keep tests whose main assertion is that retired TDD scaffolding, deleted
  methods, stale source text, or old branches are absent. Prefer positive behavior contracts and real error-path tests.

## Context for AI Assistants

- **Source Layout:** Mojo code is in `src/mo/mogemma`, Python code in `src/py/mogemma`.
- **Knowledge Base:** Curated per-component docs live in `.agents/knowledge/` (see `.agents/knowledge/README.md` for the index). Per-flow learnings live in `.agents/specs/<flow-id>/learnings.md` — never dump flow artifacts into `knowledge/`.
- **Style Guides:** `.agents/code-styleguides/python.md`, `.agents/code-styleguides/mojo.md`, `.agents/code-styleguides/testing.md`, `.agents/code-styleguides/bash.md`.
