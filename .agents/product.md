# Product

<!-- truth: start -->

## Vision

`mogemma` is a high-performance Python library for running Google's **Gemma 4**
models locally. It pairs a polished Pythonic API with pure-Mojo inference
kernels compiled into a native extension, delivering near-native throughput
without leaving the Python ecosystem.

## Target audience

- **Data scientists & ML engineers** running Gemma 4 locally for
  experimentation, evaluation, and RAG pipelines.
- **Application developers** embedding Gemma 4 in Python services and
  wanting a single-dependency install (`pip install mogemma`) with
  reliable performance characteristics.
- **Edge & research engineers** needing explicit control over memory,
  device selection, and variant loading — without a heavyweight serving
  framework.

## Core values

- **Dead simple API.** One `import mogemma`; load a model by id; call
  `.generate()` or `.embed()`. Complexity lives in the runtime, not the
  surface.
- **Native-speed execution.** Pure Mojo kernels (not a dispatcher over
  another framework) compiled into `mogemma._core` via hatch-mojo.
- **Explicit, no silent fallbacks.** Resolution, loading, and device
  selection all log and raise rather than silently degrading.
- **Multi-variant, multi-modality.** Text, image, and audio inputs across
  the Gemma 4 family.

## Shipped capabilities

- **Text generation:** `SyncGemmaModel.generate()` and
  `AsyncGemmaModel.generate()` (streaming, anyio-based).
- **Embeddings:** `SyncEmbeddingModel.embed()` and async counterpart;
  single + batched inputs, tokenized pre-input supported.
- **Vision inputs:** SigLIP-derived encoder with variable aspect-ratio
  support; patch budget respects H, W divisible by 48.
- **Audio inputs (E2B/E4B):** Mel-spectrogram preprocessing, fed to the
  same transformer structure as vision.
- **GPU acceleration:** `GPUBackend` via Mojo `std.gpu.host`; CPU-only
  builds exclude GPU symbols entirely.
- **Orbax → safetensors conversion:** Downloaded GCS checkpoints are
  converted on-the-fly to HF-style safetensors for runtime loading.

## Supported variants

Primary target is **Gemma 4** (four variants, auto-detected from `config.json`):

- `gemma-4-e2b-it` — compact multimodal (text + image + audio), PLE + double-wide MLP.
- `gemma-4-e4b-it` — larger multimodal (text + image + audio), PLE. **Library default.**
- `gemma-4-26b-a4b-it` — MoE (128 experts, top-8) with shared dense MLP; recommended for heavier reasoning workloads.
- `gemma-4-31b-it` — dense flagship, text + image.

Gemma 3 support is **legacy** — retained for downstream compatibility but
not the active development target.

## Brand & identity

- **Minimalist, professional documentation.** Short paragraphs, code-
  first, no marketing fluff.
- **Developer-experience-first CLI + APIs.** Error messages distinguish
  clearly between Python-layer and Mojo-layer failures and suggest fixes.
- **Semantic versioning.** Bumped via `bump-my-version`; release notes
  explain migration steps when needed.
<!-- truth: end -->
