# 🔥 MoGemma

Python/Mojo interface for Google Gemma 3.

## Features

- **Embeddings** — Dense vector embeddings via a pure Mojo backend.
- **Text generation** — Synchronous and async streaming with configurable sampling.
- **Multimodal Vision** — Native support for Gemma 3 Vision models with zero-copy image processing.
- **Google Cloud Storage** — Automatic model download from Google's `gemma-data` bucket.
- **OpenTelemetry** — Optional tracing instrumentation.

## Installation

```bash
pip install mogemma
```

For text generation (requires tokenizer):

```bash
pip install 'mogemma[llm]'
```

## Quick Start

### Text Generation

```python
from mogemma import SyncGemmaModel

model = SyncGemmaModel()
print(model.generate("Write a haiku about a robot discovering coffee:"))
```

### Multimodal Vision

MoGemma supports Gemma 3 multimodal vision models. Pass raw image bytes or numpy arrays to the `generate` or `generate_stream` methods.

```python
from mogemma import SyncGemmaModel
import numpy as np

# Initialize a vision-capable model
model = SyncGemmaModel("gemma3-4b-it")

# Pass an image (e.g., as a numpy array)
# images = [np.zeros((3, 384, 384), dtype=np.uint8)]

response = model.generate("Describe this image in detail:", images=images)
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
from mogemma import EmbeddingModel

model = EmbeddingModel()
embeddings = model.embed(["Hello, world!", "Mojo runs Gemma inference."])
print(embeddings.shape)  # (2, 768)
```

### Selecting a Model Variant

All model classes default to `gemma3-270m-it`. Pass a model ID to use a different variant:

```python
model = SyncGemmaModel("gemma3-1b-it")
```

For full control over sampling parameters, pass a `GenerationConfig`:

```python
from mogemma import GenerationConfig, SyncGemmaModel

config = GenerationConfig(model_path="gemma3-1b-it", temperature=0.7)
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
from mogemma import EmbeddingConfig, EmbeddingModel, GenerationConfig, SyncGemmaModel

generation = SyncGemmaModel(
    GenerationConfig(
        model_path="gemma3-1b-it",
        device="cpu",
    )
)

embeddings = EmbeddingModel(
    EmbeddingConfig(
        model_path="gemma3-1b-it",
        device="cpu",
    )
)
```

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
