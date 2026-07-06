# 🔥 MoGemma

Python/Mojo interface for Google Gemma 4.

## Features

- **Embeddings** — Dense vector embeddings via a pure Mojo backend, using the pretrained E4B variant by default.
- **Text generation** — Synchronous and async streaming with configurable sampling.
- **Multimodal** — Gemma 4 text/image runtime paths with explicit support gates for audio, MTP, thinking, and 12B unified image/audio follow-up work.
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

The default (`google/gemma-4-E4B-it`) supports the current text/image runtime path. Audio inputs are recognized but intentionally rejected until the audio runtime is implemented.

- Install `mogemma[vision]` to pass image file paths or raw image bytes directly.

```python
from mogemma import SyncGemmaModel

# Default model is google/gemma-4-E4B-it
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

Official Gemma 4 model IDs are tracked separately from runtime and GCS availability:

| Model ID | Official | GCS download | Runtime status |
|---|---:|---:|---|
| `google/gemma-4-E2B-it` | Yes | Yes | Text/image supported; audio, MTP, and thinking are follow-up work |
| `google/gemma-4-E4B-it` | Yes | Yes | **Default**; text/image supported; audio, MTP, and thinking are follow-up work |
| `google/gemma-4-12B-it` | Yes | No | Local safetensors CPU text supported; unified image/audio inputs and GPU are follow-up work |
| `google/gemma-4-31B-it` | Yes | No | Official dense model; runtime/download validation pending |
| `google/gemma-4-26B-A4B-it` | Yes | Yes | MoE path exists; live parity and full runtime validation pending |

Pretrained (non-instruction-tuned) E2B / E4B are listed in the Gemma 4 family but are not currently published to `gs://gemma-data`; `SyncEmbeddingModel` therefore defaults to the `-it` variant until pretrained ships.

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
