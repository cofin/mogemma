# Device Selection Policy

## Grammar

Supported device forms:

- `cpu`
- `gpu`
- `gpu:<index>` (for example, `gpu:0`)

Normalized backend IDs:

- `cpu` -> `cpu_mojo`
- `gpu` / `gpu:<index>` -> `gpu_mojo`

## Capability and Fallback Matrix

| Requested | GPU available | `allow_device_fallback` | Result |
| --- | --- | --- | --- |
| `cpu` | n/a | `False`/`True` | use `cpu_mojo` |
| `gpu` | `True` | `False`/`True` | use `gpu_mojo` |
| `gpu:<index>` | `True` | `False`/`True` | use `gpu_mojo` with index |
| `gpu` / `gpu:<index>` | `False` | `False` | raise deterministic runtime error |
| `gpu` / `gpu:<index>` | `False` | `True` | deterministic fallback to `cpu_mojo` |

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
    GenerationConfig(model_path="gemma3-270m-it", device="gpu:0", allow_device_fallback=False)
)

# Deterministic fallback to CPU if GPU unavailable:
fallback_gpu = SyncGemmaModel(
    GenerationConfig(model_path="gemma3-270m-it", device="gpu:0", allow_device_fallback=True)
)
```

## Manual Verification (2026-03-05)

Executed local policy checks:

- CPU-only strict mode (`gpu:0`, `allow_device_fallback=False`) -> deterministic error
- CPU-only fallback mode (`gpu:0`, `allow_device_fallback=True`) -> `('cpu_mojo', 'cpu', True)`
- Mocked GPU capability (`gpu:1`, `gpu_available=True`) -> `('gpu_mojo', 'gpu:1', False)`
