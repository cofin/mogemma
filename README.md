# 🔥 Mogemma

Python/Mojo interface for Google Gemma 3 with [MAX Engine](https://www.modular.com/max).

## Features

- **Embeddings** — Generate dense vector embeddings through the Mojo backend.
- **Text generation** — Synchronous and async streaming text generation with configurable sampling.
- **HuggingFace Hub** — Automatic model resolution and caching.
- **OpenTelemetry** — Built-in tracing instrumentation.

## Installation

```bash
pip install mogemma
```

Optional: `pip install 'mogemma[telemetry]'` for tracing.

## Quick Start

### Text Generation

```python
from mogemma import GenerationConfig, SyncGemmaModel

config = GenerationConfig(model_path="google/gemma-3-1b", max_new_tokens=64)
model = SyncGemmaModel(config)

# Full generation
print(model.generate("Explain quantum computing in one sentence:"))

# Streaming
for token in model.generate_stream("The future of AI is"):
    print(token, end="", flush=True)
```

### Async Streaming

```python
import asyncio
from mogemma import GenerationConfig, AsyncGemmaModel

async def main():
    config = GenerationConfig(model_path="google/gemma-3-1b", max_new_tokens=64)
    model = AsyncGemmaModel(config)

    async for token in model.generate_stream("Once upon a time"):
        print(token, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
```

### Embeddings

```python
from mogemma import EmbeddingConfig, EmbeddingModel

config = EmbeddingConfig(model_path="google/gemma-3-1b")
model = EmbeddingModel(config)

embeddings = model.embed(["Hello, world!", "Model outputs are computed by MAX Engine."])
print(embeddings.shape)  # (2, 768)
```

## Development

```bash
make install      # Install dependencies
make build        # Build Mojo shared library
make test         # Run tests
make lint         # Lint and type-check
```

## License

MIT
