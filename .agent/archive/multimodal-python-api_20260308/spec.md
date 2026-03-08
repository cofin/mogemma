# Flow: multimodal-python-api_20260308

## Specification

### Code Analysis Summary
**Files Analyzed**:
- `src/py/mogemma/model.py`: Contains `SyncGemmaModel.generate` and `generate_stream`, and their async equivalents. Currently only accepts text `prompt: str`.
- `src/py/mogemma/backends.py`: Defines backend protocols like `GenerationBackend` with `step` method. Missing methods for image ingestion.

### Requirements
1. **Model Interface Updates**: Extend `generate` and `generate_stream` in both `SyncGemmaModel` and `AsyncGemmaModel` to accept an optional `images` parameter (e.g. a list of byte streams or NumPy arrays).
2. **Backend Protocol Extension**: Extend `GenerationBackend` and `CoreModuleContract` in `src/py/mogemma/backends.py` with a `process_images` method.
3. **Core Integration**: Connect the Python API calls to the newly minted `process_image_mojo` from Chapter 2, ensuring that visual tokens are processed and injected into the KV cache (or prompt token sequence) before starting autoregressive text decoding.

## Implementation Plan

### Phase 1: Update Backend Contracts
- [ ] Task 1.1: Add `process_images(self, llm: object, images: Sequence[bytes | npt.NDArray]) -> None` to `CoreModuleContract` and `GenerationBackend` in `src/py/mogemma/backends.py`.
- [ ] Task 1.2: Implement `process_images` in `CoreBackend` to proxy the call to `_core.process_image_mojo`.

### Phase 2: Extend Python Models
- [ ] Task 2.1: Update `SyncGemmaModel.generate` and `generate_stream` signatures in `src/py/mogemma/model.py` to optionally accept `images`.
- [ ] Task 2.2: Within `generate_stream`, if `images` are provided, call `self._backend.process_images(self._llm, images)` right before or during the initial prompt token ingestion loop.
- [ ] Task 2.3: Mirror these signature and method changes in `AsyncGemmaModel`.