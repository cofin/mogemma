# mogemma

Python/Mojo interface for Google Gemma 3 with [MAX Engine](https://www.modular.com/max).

## Features

- **Embeddings** — Generate dense vector embeddings through the Mojo backend
- **Text generation** — Synchronous and async streaming text generation with configurable sampling (temperature, top-k, top-p)
- **HuggingFace Hub** — Automatically resolves local paths and downloads missing HF IDs into cache
- **OpenTelemetry** — Optional tracing instrumentation
- **Lazy imports** — Only loads what you use; optional extras keep the install slim

## Installation

```bash
pip install mogemma
```

### Optional extras

| Extra | What it adds | Install |
|-------|-------------|---------|
| `embed` | numpy (for embeddings) | `pip install 'mogemma[embed]'` |
| `text` | numpy + tokenizers (for text generation) | `pip install 'mogemma[text]'` |
| `hub` | huggingface-hub (for model downloading) | `pip install 'mogemma[hub]'` |
| `telemetry` | opentelemetry-api (for tracing) | `pip install 'mogemma[telemetry]'` |
| `all` | everything above | `pip install 'mogemma[all]'` |

## Quick start

### Embeddings

```python
from mogemma import EmbeddingConfig, EmbeddingModel

config = EmbeddingConfig(model_path="google/gemma-3-1b")
model = EmbeddingModel(config)

embeddings = model.embed(["Hello, world!", "Model outputs are computed by MAX Engine."])
print(embeddings.shape)  # (2, hidden_dim)
```

### Text generation

```python
from mogemma import GenerationConfig, SyncGemmaModel

config = GenerationConfig(
    model_path="google/gemma-3-1b",
    max_new_tokens=64,
    temperature=0.7,
)
model = SyncGemmaModel(config)

# Full generation
print(model.generate("Explain quantum computing in one sentence:"))

# Streaming
for token in model.generate_stream("Once upon a time"):
    print(token, end="", flush=True)
```

### Async streaming

```python
import asyncio
from mogemma import GenerationConfig
from mogemma.model import AsyncGemmaModel

config = GenerationConfig(model_path="google/gemma-3-1b", max_new_tokens=64)
model = AsyncGemmaModel(config)

async def main():
    async for token in model.generate_stream("The future of AI is"):
        print(token, end="", flush=True)

asyncio.run(main())
```

## Development

```bash
# Clone and install everything
git clone https://github.com/cofin/mogemma.git
cd mogemma
make install

# Run tests
make test

# Lint and type-check
make lint

# Build the Mojo shared library
make build
```

## License

MIT
