# Learnings for batched-embedding-contract_20260307

## [2026-03-07] - Batched Memory Layout & Python Contracts
- **Files changed:** `src/mo/mogemma/core.mojo`, `src/py/mogemma/model.py`, `src/py/tests/test_embeddings.py`
- **Learning:** Memory layout for batched embeddings requires scaling transient allocations (KV caches, scratch space) by the `batch_size`.
- **Pattern:** When allocating memory on the Mojo side, ensure `_kv_cache_len` takes the `batch_size` multiplier when dealing with embeddings, whereas standard text generation defaults to `batch_size=1` for KV caching.
- **Gotcha:** FFI to Mojo from Python requires purely rectangular arrays when passing multi-dimensional arrays (like token sequences). The Python layer must proactively check for heterogeneous sequence lengths and enforce padding before delegating to the backend. We implemented this check directly in `_embed_token_array` to throw an explicit `ValueError` if row lengths mismatch, rather than allowing a silent failure or segfault in Mojo.
