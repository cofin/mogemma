# mogemma

Python/Mojo interface for Google Gemma 3 with [MAX Engine](https://www.modular.com/max).

## Features

- **Embeddings** — Generate dense vector embeddings through the Mojo backend
- **Text generation** — Synchronous and async streaming text generation with configurable sampling (temperature, top-k, top-p)
- **HuggingFace Hub** — Automatically resolves local paths and downloads missing HF IDs into the cache
- **OpenTelemetry** — Optional tracing instrumentation
- **Lazy imports** — Only loads what you use; optional extras keep the install slim

## Installation

```bash
pip install mogemma
```

`huggingface-hub` is included in the base installation, so Hugging Face model IDs are supported by default.

### Optional extras

| Extra | What it adds | Install |
|-------|-------------|---------|
| `embed` | numpy (for embeddings) | `pip install 'mogemma[embed]'` |
| `text` | numpy + tokenizers (for text generation) | `pip install 'mogemma[text]'` |
| `telemetry` | opentelemetry-api (for tracing) | `pip install 'mogemma[telemetry]'` |
| `all` | everything above | `pip install 'mogemma[all]'` |

### Model resolution behavior

- Model identifiers are accepted in `namespace/model` format (for example, `google/gemma-3-1b`).
- Missing IDs are downloaded into the local cache on first use unless disabled by your offline settings.

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

# Run release preflight checks
make check-release

# Build the Mojo shared library
make build
```

## Troubleshooting

- `Model path ... exists but is not a directory`: point to a model directory or use a valid Hugging Face model id.
- `offline mode`: set `HF_HUB_OFFLINE=0`, ensure network access, then retry.
- `HF_TOKEN`: configure a token (`huggingface-cli login`) for private/restricted repos.
- Cached downloads are stored in `~/.cache/mogemma`.

## Implementation evidence

- Core output contracts and error paths:
  - `src/mo/tests/test_core_contract.py`
  - `src/py/tests/test_contracts_integration.py`
- Runtime contract behavior (sync/async generation and embeddings):
  - `src/py/tests/test_gemma_model.py`
  - `src/py/tests/test_embeddings.py`
  - `src/py/tests/test_async.py`
- Model delivery and hub error handling:
  - `src/py/tests/test_hub.py`

## License

MIT
