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

## Migration Strategy

1. Phase A (this chapter):
   - Introduce backend protocols and resolver seam.
   - Keep CPU behavior intact through `CPUCoreBackend`.
2. Phase B (GPU adapter introduction):
   - Add `gpu_mojo` adapter implementations behind resolver branches.
   - Keep `cpu_mojo` as deterministic fallback until parity gates pass.
3. Phase C (runtime/device integration):
   - Integrate runtime-state and device-selection flows into the adapter seam.
   - Avoid public API changes unless required for explicit capability reporting.
4. Phase D (validation + promotion):
   - Enforce parity and performance gates before changing backend defaults.
   - Keep rollout policy tied to documented release and runbook guidance.

## Chapter Handoff Map (PRD1 Checkpoint)

### Handoff to Chapter 2 (`gpu-runtime-state-refactor_20260304`)

1. `src/py/mogemma/model.py`:
   - Stable seam: model-level cached backend fields (`_backend`, `_backend_id`, `_llm`).
   - Chapter 2 can optimize runtime/session lifecycle without changing public APIs.
2. `src/py/mogemma/backends.py`:
   - Stable seam: protocol methods (`init_model`, `step`, `generate_embeddings`) and resolver entrypoints.
   - Chapter 2 can introduce richer runtime state objects as long as adapter contracts stay intact.

### Handoff to Chapter 3 (`gpu-device-selection-api_20260304`)

1. `src/py/mogemma/backends.py`:
   - Stable seam: `resolve_backend_id` already normalizes `cpu|gpu|gpu:N`.
   - Chapter 3 should extend this into full device-capability semantics and fallback policy.
2. `src/py/mogemma/config.py` + `src/py/mogemma/model.py`:
   - Stable seam: `config.device` is threaded into backend resolution.
   - Chapter 3 can add richer validation/capability errors while preserving current defaults.
