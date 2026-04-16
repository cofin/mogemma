# Python runtime (`src/py/mogemma/`)

## `hub.py` — HubManager

- Cache root: `$MOGEMMA_CACHE_DIR` or `~/.cache/mogemma`.
- GCS bucket: `gemma-data`, public, `skip_signature=true`. Config typed as `GCSConfig | None` in obstore stubs but plain dict works at runtime — cast via `config: Any = {...}`.
- Async concurrency: `_ASYNC_DOWNLOAD_CONCURRENCY = 6` for GCS downloads.
- Model-id normalization: strip `google/`, `gemma-` → `gemma`, lowercase.
- HF path uses obstore `HTTPStore` (not the official `huggingface_hub` client — avoids extra dep).
- HF doesn't expose bucket listing: must read `model.safetensors.index.json` to enumerate shards.
- Staging pattern: download to `staging_dir`, validate, `staging_dir.rename(local_dir)` atomically.
- Post-download hook: `_finalize_download` converts Orbax → safetensors if needed.

## `orbax_loader.py`

- Wraps TensorStore / OCDBT. Full `OrbaxLoader()` eager-loads all tensors — DO NOT use for multi-GB checkpoints.
- **Streaming helpers** for conversion path:
  ```python
  OrbaxLoader.enumerate_tensors(path)        # list[str], no data materialized
  OrbaxLoader.open_tensor(path, name)        # single np.ndarray, bf16 → f32 promoted
  ```
- Internal stub pattern: `cls.__new__(cls)` skips eager init, assigns `model_path` manually, calls private `_enumerate_tensor_names()` / `_open_tensor()`.
- bfloat16 is converted to float32 at open time (numpy has no native bf16; TensorStore decodes via left-shift).
- Returned arrays are made C-contiguous before return.

## `safetensors_loader.py`

- Multi-file shards: parse `model.safetensors.index.json`, map tensor names to shard files.
- Zero-copy: `safetensors.safe_open(file, framework="numpy")` → mmap. Keep file handle alive for the lifetime of any pointer handed to Mojo.
- Shape contract is the HF-normalized layout (already-transposed, already-split), not Orbax packing.

## `convert.py` — Orbax → safetensors

See [gemma4-models.md](gemma4-models.md) for Orbax tensor layouts. This module
transforms them to HF-style safetensors the runtime can load.

- Streaming single-pass writer `_write_sharded(output_dir, tensor_iter, shard_size_bytes)`:
  unknown total shard count up-front → write with temp names, rename at end.
- Variant detection: `_variant_from_keys(keys) -> "base" | "ple" | "moe"`.
- Per-variant iterators: `_iter_base_transformer`, `_iter_ple`, `_iter_moe_transformer`, `_iter_vision`.
- `_generate_config_json(keys, shape_oracle)` synthesizes HF config if GCS didn't ship one.
- Experts stay **packed** (`[E, 2·I_moe, H]`) rather than per-expert split — matches HF `Gemma4TextExperts.gate_up_proj`.

## `config.py`

- `GenerationConfig`: `max_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, etc. Zero-temperature path must be deterministic.
- `EmbeddingConfig`: pooling strategy, model variant.
- Explicit typing throughout — no `Any` in public API.

## `model.py` — public API

- Classes: `SyncGemmaModel`, `AsyncGemmaModel`, `SyncEmbeddingModel`, `AsyncEmbeddingModel`.
- Lazy tokenizer init: fail-fast on first `.generate()` if sentencepiece model is missing.
- Error taxonomy (explicit, never silent):
  - `ModelNotFoundError`
  - Tokenizer-missing error
  - `_core` (Mojo extension) missing error
  - Runtime error (from FFI)
- Async wrappers use `anyio` (not asyncio-specific) so they work under trio too.
- `test_async_generate_stream` is **flaky** — `assert len(tokens) > 0` fails intermittently; pre-existing, not a recent regression.

## `loader.py` — model resolution

Resolution precedence (explicit, documented):

1. Local path (absolute or relative directory containing `config.json`).
2. Local cache (`$MOGEMMA_CACHE_DIR`).
3. Remote download via `HubManager`.

No silent fallback — each miss logs the miss and moves to the next tier.

## `audio.py`

- Computes log-mel spectrograms for audio inputs (E2B/E4B).
- Float32 output tensors, passed to Mojo via FFI.

## `backends.py`

- CPU/GPU routing for Python-side operations. Runtime backend selection based on `_gpu_initialized` flag in `llm` dict.

## Config lifecycle (what config keys flow where)

1. `config.json` is read by `model.py:__init__`.
2. Required keys validated at `model.py:109-203` — see [gemma4-models.md](gemma4-models.md#config-keys-consumed-by-mojo) for the full list.
3. Config values passed as scalars to `_core.init_model(...)`.
4. Mojo stores them in the `llm` dict.

## Tokenizer

- sentencepiece model file: `tokenizers/tokenizer_gemma4.model` (GCS path).
- Shared across all Gemma 4 variants.
- Tokenizer padding is **disabled** — Mojo does dynamic padding to max sequence length inside the runtime (better perf than Python-side padding).
