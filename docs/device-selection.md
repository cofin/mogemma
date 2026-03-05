# Device Selection Policy

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
| `gpu` | `True` | use `gpu` |
| `gpu:<index>` | `True` | use `gpu` with index |
| `gpu` / `gpu:<index>` | `False` | raise deterministic runtime error |

## Capability Hook

Current GPU capability probe is environment-based:

- `MOGEMMA_GPU_AVAILABLE=1|true|yes|on` => GPU available
- otherwise => treated as unavailable

This keeps behavior deterministic in tests and CPU-only hosts until runtime GPU probing is implemented.

## Examples

```python
from mogemma import GenerationConfig, SyncGemmaModel

# Deterministic error if GPU is unavailable
gpu_model = SyncGemmaModel(
    GenerationConfig(model_path="gemma3-270m-it", device="gpu:0")
)
```

## Manual Verification (2026-03-05)

Executed local policy checks:

- CPU-only host (`gpu:0`) -> deterministic error
- Mocked GPU capability (`gpu:1`, `gpu_available=True`) -> selected `('gpu', 'gpu:1')`
