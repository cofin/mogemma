## [2026-03-06 00:40] Revision 1

**Type:** both
**Reason:** Tightened the KV-share/sparsity chapter so implementation cannot drift on cache ABI, failure semantics, or hot-path memory behavior.
**Changes:** Added a cache ABI contract, exact remap/write invariants, allocation/sync budgets, stronger long-sequence/reset tests, and explicit copy-count/perf evidence requirements.
