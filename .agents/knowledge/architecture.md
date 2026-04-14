# System Architecture

## Package layout

```
mogemma/
├── src/py/mogemma/        # Python package (user-facing API)
│   ├── __init__.py        # lazy __getattr__ exports
│   ├── hub.py             # HubManager — GCS + HF download
│   ├── orbax_loader.py    # TensorStore/OCDBT reader (streaming helpers)
│   ├── safetensors_loader.py
│   ├── convert.py         # Orbax → safetensors conversion
│   ├── config.py          # GenerationConfig, EmbeddingConfig
│   ├── model.py           # Sync/AsyncGemmaModel, SyncEmbeddingModel
│   ├── loader.py          # Model resolution (path > cache > remote)
│   ├── audio.py           # Mel-spectrogram preprocessing
│   ├── backends.py        # CPU/GPU routing
│   └── _core.so           # hatch-mojo compiled Mojo extension (build artifact)
├── src/py/tests/          # pytest suite
├── src/mo/mogemma/        # Mojo sources
│   ├── core.mojo          # FFI entry points, step_mojo dispatch
│   ├── layers.mojo        # Forward-pass kernels (attention, MLP, MoE, vision)
│   ├── model.mojo         # TensorInfo + *LayerWeights + *ModelWeights structs
│   ├── gpu_context.mojo   # WeightStage, GPU packers, GPUContext
│   ├── ops.mojo           # ComputeBackend trait + CPUBackend
│   └── ops_gpu.mojo       # GPUBackend (guarded by has_accelerator())
└── src/mo/tests/          # Mojo test suite
```

## Python ↔ Mojo bridge

- hatchling + `hatch-mojo>=0.1.8` compiles `src/mo/mogemma/core.mojo` → `mogemma._core.so`.
- Python imports via `mogemma._core`. The build artifact is dropped into `src/py/mogemma/`.
- FFI marshalling: Python passes numpy pointers as integers. Mojo reconstructs via `UnsafePointer[T, MutExternalOrigin](unsafe_from_address=Int(py=...))`.
- Ragged tensors (embeddings, token batches) are padded to rectangular in Python before crossing FFI — numpy will not infer jagged shapes, and Mojo kernels require contiguous buffers.
- **GC protection:** Python retains references to numpy arrays whose pointers Mojo holds. Premature GC corrupts Mojo weights.

## Module boundaries

| Layer | Owns | Does NOT own |
|---|---|---|
| `hub.py` | Network I/O, cache layout, model-id normalization | Weight format interpretation |
| `orbax_loader.py` / `safetensors_loader.py` | Reading a checkpoint into numpy arrays | Mojo struct layout |
| `convert.py` | Orbax → safetensors transform (shape/transpose/split) | Download, tokenizer, runtime |
| `model.py` | Public API (`generate`, `embed`), lifecycle, async wrappers | Forward-pass math |
| Mojo `core.mojo` | FFI entry points, dispatch between CPU/GPU, session state | Python-side config parsing |
| Mojo `layers.mojo` | Forward-pass math, backend-agnostic kernels | Weight loading, staging |
| Mojo `gpu_context.mojo` | GPU memory orchestration, weight packing for upload | CPU fallback logic |

## Session/runtime state

- `llm` dict (Python) holds opaque handles: `pos`, `session_kv_cache_len`, `step_scratch_len`, `embedding_scratch_len`, and GPU buffer pointers as integers (heap pointers cast to `Int`).
- Mojo side retrieves with `UnsafePointer(unsafe_from_address=Int(py=llm["_key"]))`.
- `_gpu_initialized` flag in the llm dict controls CPUBackend vs GPUBackend dispatch in `step_mojo`.

## Data flow (text generation)

```
user.generate(prompt)
  → SyncGemmaModel.generate                  # model.py
  → tokenizer.encode                         # sentencepiece
  → mogemma._core.step_mojo(ptrs, config)    # FFI
  →   core.mojo: _run_step[Backend, KVCache] # backend latch
  →   layers.mojo: forward_layer × N         # backend-agnostic
  →     backend.rms_norm / vec_mat_mul / ... # dispatch via trait
  → sampler (Python or Mojo)                 # depending on config
  → tokenizer.decode                         # back to text
```

## Related

- Build details: [build-and-packaging.md](build-and-packaging.md)
- GPU memory model: [gpu-infrastructure.md](gpu-infrastructure.md)
- Struct layouts: [mojo-runtime.md](mojo-runtime.md)
