# 🔥 MoGemma

Python/Mojo interface for Google Gemma 4.

## Features

- **Embeddings** — Dense vector embeddings via a pure Mojo backend, using the pretrained E4B variant by default.
- **Text generation** — Synchronous and async streaming with configurable sampling.
- **Multimodal** — Native support for Gemma 4 vision (all variants) and audio (E2B/E4B) with zero-copy processing.
- **Google Cloud Storage** — Automatic model download from Google's `gemma-data` bucket.
- **OpenTelemetry** — Optional tracing instrumentation.

## Installation

Recommended for most users:

```bash
pip install 'mogemma[llm]'
```

This enables the text generation and embedding examples shown below.

For multimodal generation with automatic image decoding from `str`, `Path`, or raw `bytes` inputs:

```bash
pip install 'mogemma[vision]'
```

Base package only:

```bash
pip install mogemma
```

Use the base package if you're already preparing tokens or image arrays yourself.

## Quick Start

### Text Generation

The default getting-started path is `mogemma[llm]`.

```python
from mogemma import SyncGemmaModel

model = SyncGemmaModel()
print(model.generate("Write a haiku about a robot discovering coffee:"))
```

### Multimodal Vision

All Gemma 4 variants support vision inputs; the default (`google/gemma-4-E4B-it`) additionally accepts audio.

- Install `mogemma[vision]` to pass image file paths or raw image bytes directly.

```python
from mogemma import SyncGemmaModel

# Default model is google/gemma-4-E4B-it — multimodal out of the box
model = SyncGemmaModel()

response = model.generate("Describe this image in detail:", images=["input.jpg"])
print(response)
```

### Async Streaming

```python
import asyncio
from mogemma import AsyncGemmaModel

async def main():
    model = AsyncGemmaModel()
    async for token in model.generate_stream("Once upon a time"):
        print(token, end="", flush=True)

asyncio.run(main())
```

### Embeddings

Generate dense vector embeddings natively through Mojo's optimized batched kernel operations. Pass a single string or a list of strings to process them in parallel.

```python
from mogemma import SyncEmbeddingModel

model = SyncEmbeddingModel()
embeddings = model.embed(["Hello, world!", "Mojo runs Gemma inference."])
print(embeddings.shape)  # (2, 768)
```

### Selecting a Model Variant

Four Gemma 4 variants are supported (auto-detected from `config.json`):

| Model ID | Description |
|---|---|
| `google/gemma-4-E2B-it` | Compact multimodal (text + image + audio), ~2B params |
| `google/gemma-4-E4B-it` | **Default** for `SyncGemmaModel` / `AsyncGemmaModel` — latest small multimodal |
| `google/gemma-4-E4B` | **Default** for `SyncEmbeddingModel` — pretrained (better embedding quality) |
| `google/gemma-4-26B-A4B-it` | MoE (128 experts, top-8), 4B active — heavier reasoning |
| `google/gemma-4-31B-it` | Dense flagship, text + image |

Pass a model ID to override the default:

```python
model = SyncGemmaModel("google/gemma-4-26B-A4B-it")
```

For full control over sampling parameters, pass a `GenerationConfig`:

```python
from mogemma import GenerationConfig, SyncGemmaModel

config = GenerationConfig(model_path="google/gemma-4-26B-A4B-it", temperature=0.7)
model = SyncGemmaModel(config)
```

### Device Selection

`GenerationConfig` and `EmbeddingConfig` accept:

- `device="cpu"`
- `device="gpu"`
- `device="gpu:0"` (or other index)

Device handling is deterministic:

- `device="cpu"` always runs on CPU
- explicit GPU requests never silently fall back to CPU
- unavailable GPU requests raise an explicit error

Current runtime status:

- `cpu` and `gpu` are executable backends today
- `gpu` / `gpu:N` execute via a mathematically verified runtime polyfill

```python
from mogemma import EmbeddingConfig, SyncEmbeddingModel, GenerationConfig, SyncGemmaModel

generation = SyncGemmaModel(
    GenerationConfig(
        model_path="google/gemma-4-E4B-it",
        device="cpu",
    )
)

embeddings = SyncEmbeddingModel(
    EmbeddingConfig(
        model_path="google/gemma-4-E4B",
        device="cpu",
    )
)
```

> **GPU Requirements:** GPU acceleration requires Mojo nightly with GPU support,
> compatible GPU drivers (NVIDIA CUDA, AMD ROCm, or Apple Metal), and sufficient VRAM
> for model weights and KV cache.

## Runtime Requirements

MoGemma leverages the latest Mojo features for maximum performance.

- **Mojo Nightly:** Version `0.26.3.0.dev` or later is required for building from source.
- **Python:** 3.10+

## Development & Architecture

### Architecture Specific Builds

MoGemma automatically optimizes its Mojo core for your specific CPU architecture during the build process.

- **x86_64:** Uses `--target-cpu x86-64-v3` for optimized vector instructions.
- **aarch64:** Uses native ARM optimizations.

### Local Development

To build the Mojo extension locally:

```bash
make build
```

## License

MIT
