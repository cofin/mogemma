# 🔥 Mogemma

Python/Mojo interface for Google Gemma 3 with [MAX Engine](https://www.modular.com/max).

## Features

- **Embeddings** — Generate dense vector embeddings through the Mojo backend.
- **Text generation** — Synchronous and async streaming text generation with configurable sampling.
- **Google Cloud Storage** — Automatic model resolution and high-performance concurrent downloading directly from Google's `gemma-data` bucket.
- **OpenTelemetry** — Built-in tracing instrumentation.

## Installation

```bash
pip install mogemma
```

Optional: `pip install 'mogemma[llm]'` for generation support (numpy + tokenizers + modular).

**Note**: Text generation requires the **Modular MAX Engine** (nightly build):
```bash
pip install modular
```

## Quick Start

### Text Generation

Requires `pip install 'mogemma[llm]'`.

```python
from mogemma import GenerationConfig, SyncGemmaModel

# Defaults to gemma3-270m-it
config = GenerationConfig(max_new_tokens=64)
model = SyncGemmaModel(config)

# Full generation
print(model.generate("Explain quantum computing in one sentence:"))

# Streaming
for token in model.generate_stream("The future of AI is"):
    print(token, end="", flush=True)
```

### Async Streaming

Requires `pip install 'mogemma[llm]'`.

```python
import asyncio
from mogemma import GenerationConfig, AsyncGemmaModel

async def main():
    # Defaults to gemma3-270m-it
    config = GenerationConfig(max_new_tokens=64)
    model = AsyncGemmaModel(config)

    async for token in model.generate_stream("Once upon a time"):
        print(token, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
```

### Embeddings

```python
from mogemma import EmbeddingConfig, EmbeddingModel

# Defaults to gemma3-270m-it
config = EmbeddingConfig()
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
