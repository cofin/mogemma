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

| Requested | GPU available | `unavailable_gpu_policy` | Result |
| --- | --- | --- | --- |
| `cpu` | n/a | `error`/`use_cpu` | use `cpu` |
| `gpu` | `True` | `error`/`use_cpu` | use `gpu` |
| `gpu:<index>` | `True` | `error`/`use_cpu` | use `gpu` with index |
| `gpu` / `gpu:<index>` | `False` | `error` | raise deterministic runtime error |
| `gpu` / `gpu:<index>` | `False` | `use_cpu` | deterministic CPU downgrade |

## Capability Hook

Current GPU capability probe is environment-based:

- `MOGEMMA_GPU_AVAILABLE=1|true|yes|on` => GPU available
- otherwise => treated as unavailable

This keeps behavior deterministic in tests and CPU-only hosts until runtime GPU probing is implemented.

## Examples

```python
from mogemma import GenerationConfig, SyncGemmaModel

# Deterministic error if GPU is unavailable:
strict_gpu = SyncGemmaModel(
    GenerationConfig(model_path="gemma3-270m-it", device="gpu:0", unavailable_gpu_policy="error")
)

# Deterministic CPU downgrade if GPU unavailable:
cpu_downgrade_gpu = SyncGemmaModel(
    GenerationConfig(model_path="gemma3-270m-it", device="gpu:0", unavailable_gpu_policy="use_cpu")
)
```

## Manual Verification (2026-03-05)

Executed local policy checks:

- CPU-only strict mode (`gpu:0`, `unavailable_gpu_policy="error"`) -> deterministic error
- CPU-only downgrade mode (`gpu:0`, `unavailable_gpu_policy="use_cpu"`) -> `('cpu', 'cpu', True)`
- Mocked GPU capability (`gpu:1`, `gpu_available=True`) -> `('gpu', 'gpu:1', False)`
