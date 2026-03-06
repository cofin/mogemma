## [2026-03-06 00:28] Revision 1

**Type:** both
**Reason:** The original plan did not pin down ownership, sync budgets, or steady-state allocation rules tightly enough for a high-confidence implementation PR.
**Changes:** Added explicit touch points, memory/sync invariants, byte-budget and allocation constraints, deterministic failure semantics, clearer generation-vs-embedding boundaries, and a more exact TDD/validation sequence for cache residency and session reuse.
