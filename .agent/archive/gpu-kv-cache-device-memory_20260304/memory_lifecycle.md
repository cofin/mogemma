# Phase 1: Device Memory Contract for Standard Runtime

## 1.1 Standard Session-Memory Model

*   **Long-lived Device KV Buffers:** `k_cache` and `v_cache` will transition from Python-allocated `numpy.zeros` arrays on the host to backend-allocated opaque handles (in GPU sessions) or remain host arrays (in CPU sessions). They are sized to `[num_layers, max_seq_len, num_kv_heads, head_dim]` and persist across generation tokens.
*   **Long-lived Step Scratch:** A single, contiguous `step_scratch` buffer is allocated once per session. For GPU sessions, this lives on the device. `step_mojo` will stop allocating `_allocate_transient_f32` per token.
*   **Embedding Scratch Pools:** `generate_embeddings_mojo` retains its current host-transient allocations to avoid regressing generation-session reuse, as the sequence length varies unpredictably. This may be optimized in a later flow.
*   **Host-Visible Staging Buffers:**
    *   **Input Staging:** Token IDs are naturally passed via Python `int`.
    *   **Logits Egress:** A 1D host array (`out_logits`) will still be allocated per step in `step_mojo` for returning to the Python caller.

## 1.2 Python `llm` Visibility & Opaque Backend Handles

The Python API's expected contract relies on the `llm` dictionary being a transparent state container for resetting and probing.

**Visible fields (unchanged or newly tracked):**
*   `pos`: Current token sequence position. Resetting this to 0 clears the cache logically.
*   `session_kv_cache_len`: Total elements per cache.
*   `step_scratch_len`: Total elements for step scratch.
*   `embedding_scratch_len`: Kept for embedding queries.
*   `step_backend` & `fallback_reason`: Exposes active backend status.

**Opaque Handles (Backend-managed):**
*   `k_cache` and `v_cache` will now hold custom Python objects or opaque integer handles when `step_backend == "cuda"`. Python must *not* assume `__array_interface__` is readable for these entries unless `step_backend == "cpu"`.

## 1.3 Transfer & Sync Matrix

| Operation Phase | Action | Source | Destination | Sync Requirement |
| --- | --- | --- | --- | --- |
| Session Init | Allocate KV & Scratch | Host | Device (GPU backend) | Synchronous wait for allocation success. |
| Token Upload | Send `token_id` | Host | Device | Asynchronous or blocking before compute kernel dispatch. |
| Compute | Run Kernels | Device | Device | Pipelined asynchronously on stream. Cache resides on device entirely. |
| Logits Egress | Transfer `out_logits` | Device | Host | **One explicit sync point per step** before returning logits to Python. |
| Reset | Logical Cache Clear | Host | Device | No transfer. Host sets `llm["pos"] = 0`. Device compute relies on `pos`. |
| Teardown | GC/Free | Host | Device | Managed by Python garbage collection on the opaque handles (or explicit destructors). |
| Fallback | CPU Session | Host | Host | Synchronous. `__array_interface__` pointers extracted directly. |

## 1.4 Lifecycle Diagram (Approved Checkpoint)

```text
[INIT_MODEL] -> Evaluates Device -> Allocates Opaque Handles -> Returns `llm` Dict
                     |
                     v
[STEP (pos=0)] -> Checks `llm["step_backend"]` 
                     |---> (CPU Fallback) Extract `numpy` pointers -> Run CPU Ops -> Return Logits
                     |---> (GPU Dispatch) Pass Handle + `token_id` -> Enqueue Kernels -> Sync -> Return Host Logits
                     |
                     v
[STEP (pos=1)] -> Reuses `llm["step_backend"]` and Handles
                     |
                     v
[SESSION RESET] -> Python sets `llm["pos"] = 0` (Handles remain alive, data overwritten safely by write-before-attend)
                     |
                     v
[SESSION END] -> `del llm` -> Opaque handles destroyed -> Device memory freed
```
