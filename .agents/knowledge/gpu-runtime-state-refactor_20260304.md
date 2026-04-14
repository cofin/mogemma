# Learnings: gpu-runtime-state-refactor_20260304

## 2026-03-05

- Descriptor-cache instrumentation must be validated against the built artifact (`src/py/mogemma/_core.so`), not only source edits; run `make build` before asserting Mojo runtime counter behavior in Python tests.
- Mojo cannot persist non-Python structs inside `Python.dict`, so hot-path nano caching in `step_mojo` should use runtime-layer tensor views plus per-layer reconstruction, not direct `NanoModelWeights` object storage.
- Reusing one runtime-parse path for both `step_mojo` and `generate_embeddings_mojo` keeps nano cache semantics aligned and prevents regressions where one path silently reintroduces full model builds.
- A practical KV-share invariant check is cache-slot based: force non-zero writes in the last full-KV layer, then assert shared-layer cache slices remain all-zero after `step()`.
- Consecutive `step -> step -> generate_embeddings -> step` checks are a low-cost stability guard for session-state regressions (`pos` must remain unchanged by embedding calls).
- Session reset must handle list-backed cache containers explicitly; `np.asarray(list).fill(0)` mutates a copy, not the underlying Python list.
- Keeping explicit allocation-size metadata on the session object (`session_kv_cache_len`, `step_scratch_len`, `embedding_scratch_len`) makes long-lived vs transient buffer ownership auditable and keeps hot-path allocation formulas centralized.
