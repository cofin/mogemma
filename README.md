# 🔥 Mogemma

Python/Mojo interface for Google Gemma 3.

## Features

- **Embeddings** — Dense vector embeddings via a pure Mojo backend.
- **Text generation** — Synchronous and async streaming with configurable sampling.
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
print(model.generate("Explain quantum computing in one sentence:"))
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

## License

MIT
