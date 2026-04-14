# Hard-won learnings

Cross-cutting gotchas that don't belong to a single component. Keep this
curated — delete entries when they're no longer true; move to the relevant
component file if they become scope-specific.

## FFI and memory

- **GC corruption**: Python `numpy.ndarray` whose pointer crossed into Mojo must stay referenced on the Python side for the model's lifetime. Premature GC corrupts weights mid-forward with no crash — just garbage output.
- **bf16 handling**: numpy has no native bf16. TensorStore decodes by left-shifting 16 bits into fp32. Keep the source array alive until Mojo has consumed it (same GC protection).
- **Alignment**: raw-pointer casts need 64-bit alignment. Some TensorStore arrays need `np.ascontiguousarray()` before pointer extraction.
- **Multi-file safetensors**: require `model.safetensors.index.json` parsing — HF doesn't expose bucket listing. Keep the `safe_open` file handle alive for the lifetime of any pointer handed to Mojo (mmap dies with the handle).

## Padding

- `numpy.asarray` on jagged lists fails with "inhomogeneous shape". Pad ragged batches to rectangular in Python before FFI.
- Tokenizer padding is **disabled** — Mojo does dynamic padding to max-seq-len internally (better perf than Python padding).

## Async semantics

- Use `anyio`, not `asyncio`-specific primitives — the test suite covers trio too.
- Cancellation is environment-sensitive: what works in asyncio may silently hang in trio. Prefer `anyio.CancelScope` over raw signals.
- Never swallow `CancelledError`.

## Silent fallbacks are bugs

- Eliminate implicit fallbacks anywhere on the public path. They mask real issues and turn deterministic failures into heisenbugs.
- When adding a tier system (e.g., resolver's local > cache > remote), log every miss. No silent retries.
- Placeholder stubs that "pass" tests via default return are a footgun — use deterministic fixtures for parity testing instead.

## Tooling traps

- **conda-forge `libstdcxx-ng`**: empty transitional package. Use `libstdcxx` (no `-ng`).
- **gcc-toolset-13** on AlmaLinux 9: no runtime `libstdc++.so.6`, only static libs + headers.
- **cibuildwheel skip identifiers**: underscore between arch components (`*_i686`), not hyphen.
- **cibuildwheel before-all**: runs **inside** the Docker container on Linux; macOS doesn't use it.
- **auditwheel repair**: refuses wheels with newer GLIBCXX symbols. Since hatch-mojo already handles bundling, use `wheel tags` to retag without repair.

## Mojo-specific

- Lazy init that requires a dependency (e.g., sentencepiece model) introduces non-deterministic failures. Validate dependency state upfront, at construct time.
- Zero-temperature paths must be deterministic; stochastic branch coverage required separately.
- `@comptime if has_accelerator()` is the gate for GPU code. CPU-only builds will see zero GPU symbols.
- `rebind` inside GPU branches casts backend-erased types back to concrete `WeightStage` / `GPUContext`.
- Scratch arithmetic must stay within a **single** `DeviceBuffer` — multiple buffers would require `enqueue_copy` which breaks the pointer-math pattern.

## Benchmark reality

- Hardware-specific perf tuning creates false baselines on other platforms. Always treat perf thresholds as advisory unless explicitly labeled mandatory (= correctness).
- Don't gate PRs on perf — latency regressions comment, correctness regressions block.

## Dependency management

- Moving a dep optional → base changes reproducibility. Refresh lockfile, validate in clean env.
- Dead-code audit after policy/dependency changes — stale imports masquerade as passing.
- `tomllib` landed in 3.11. For 3.10 support use `tomli` via conditional import.

## Contract discipline

- Lock public contracts (function signatures, Mojo struct layouts, FFI data shapes) **before** implementation. Changing a struct mid-implementation propagates across Python loader + Mojo packer + GPU uploader + tests.
- When Gemma architecture inventories contradict a spec, update the spec first and regenerate derived code — don't patch code around a wrong spec.

## Scope discipline

- When a task turns out bigger than expected, refine the plan or ask the user how to prioritize. **Never silently descope** — stubbing with `NotImplementedError` hides work and creates misleading "completion" signals.
- Placeholder returns that make tests pass are a form of scope cutting.

## Known flaky test

- `test_async_generate_stream` — `assert len(tokens) > 0` fails intermittently. Pre-existing. Do not block on it.
