# Backend Architecture and Migration Notes

## Current Runtime Shape

- Python model classes (`SyncGemmaModel`, `EmbeddingModel`) resolve a backend and cache it at construction time.
- `cpu_mojo` is implemented by `CPUCoreBackend`, which delegates into `mogemma._core` (`init_model`, `step`, `generate_embeddings`).
- Backend IDs are normalized in one place (`resolve_backend_id`) and support:
  - `cpu`, `cpu_mojo`
  - `gpu`, `gpu_mojo`, `gpu:<index>` (identifier-level support only in this chapter)

## Future GPU Backend Hooks (No Kernels Yet)

The following hooks are intentionally left as extension points for `gpu_mojo`:

1. `resolve_generation_backend()` in `src/py/mogemma/backends.py`:
   - Add `gpu_mojo` branch returning a GPU generation adapter that matches the `GenerationBackend` protocol.
2. `resolve_embedding_backend()` in `src/py/mogemma/backends.py`:
   - Add `gpu_mojo` branch returning a GPU embedding adapter that matches the `EmbeddingBackend` protocol.
3. `CPUCoreBackend` contract surface:
   - Keep adapter method names stable (`init_model`, `step`, `generate_embeddings`) so GPU adapters can be drop-in replacements.
4. Model wiring in `src/py/mogemma/model.py`:
   - `self._backend` / `self._backend_id` are the stable runtime seam for Chapter 2 and Chapter 3 work.

## Migration Safety Constraints

- Do not reintroduce MAX as the primary runtime path.
- Preserve existing public model APIs while swapping backend implementations.
- Keep backend resolution deterministic and cheap (one-time resolution per model instance).
