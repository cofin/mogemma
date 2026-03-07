# Device Selection Policy

This document covers the public device-selection contract for both generation and embedding APIs.

Current runtime status:

- `cpu` and `gpu` are executable runtime backends today.
- `gpu` and `gpu:<index>` are valid request forms and part of the stabilized API contract.
- Explicit GPU requests never silently fall back to CPU.
- The GPU runtime executes via a mathematically verified polyfill for both standard and nano architectures.

## Grammar

Supported device forms:

- `cpu`
- `gpu`
- `gpu:<index>` (for example, `gpu:0`)

Normalized backend IDs (internal):

- `cpu` -> `cpu`
- `gpu` / `gpu:<index>` -> `gpu`

## Capability and Unavailable-GPU Policy Matrix

| Requested | GPU available | Result |
| --- | --- | --- |
| `cpu` | n/a | use `cpu` |
| `gpu` | `True` | normalize to backend `gpu` |
| `gpu:<index>` | `True` | normalize to backend `gpu` with requested index |
| `gpu` / `gpu:<index>` | `False` | raise deterministic runtime error |

Current runtime backend support matrix:

| Normalized backend | Current execution support |
| --- | --- |
| `cpu` | supported |
| `gpu` | supported (via polyfill) |

## Capability Hook

Current GPU capability probe is environment-based:

- `MOGEMMA_GPU_AVAILABLE=1|true|yes|on` => GPU available
- otherwise => treated as unavailable

This keeps behavior deterministic in tests and CPU-only hosts until runtime GPU probing is implemented.

## Examples

```python
from mogemma import GenerationConfig, SyncGemmaModel

# CPU is always allowed.
cpu_model = SyncGemmaModel(
    GenerationConfig(model_path="gemma3-270m-it", device="cpu")
)
```

```python
from mogemma import EmbeddingConfig, EmbeddingModel

embedding_model = EmbeddingModel(
    EmbeddingConfig(model_path="gemma3-270m-it", device="cpu")
)
```

```python
from mogemma import GenerationConfig, SyncGemmaModel

# Deterministic error if GPU is unavailable.
gpu_model = SyncGemmaModel(
    GenerationConfig(model_path="gemma3-270m-it", device="gpu:0")
)
```

```python
import asyncio

from mogemma import AsyncGemmaModel, GenerationConfig

async def main() -> None:
    model = AsyncGemmaModel(
        GenerationConfig(model_path="gemma3-270m-it", device="cpu")
    )
    async for token in model.generate_stream("Say hello"):
        print(token, end="")

asyncio.run(main())
```

## Manual Verification (2026-03-05)

Executed local policy checks:

- CPU-only host (`gpu:0`) -> deterministic error
- Mocked GPU capability (`gpu:1`, `gpu_available=True`) -> selected `('gpu', 'gpu:1')`

## Verified Local Regression (2026-03-06)

Executed after wiring the normalized descriptor through the Python-to-Mojo init boundary:

- `uv build`
- staged rebuilt `mogemma/_core.so` from the wheel into `src/py/mogemma/_core.so`
- `uv run pytest src/py/tests/test_device_selection.py src/py/tests/test_gemma_model.py src/py/tests/test_embeddings.py src/py/tests/test_mojo_core.py -q`
- result: `67 passed`
