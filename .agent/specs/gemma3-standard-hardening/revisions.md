## [2026-03-06 00:28] Revision 1

**Type:** both
**Reason:** Tightened the active hardening spec so the remaining implementation work is fail-fast, performance-safe, and specific enough for a single-pass PR.

### Changes Made

**Spec Changes:**
- Added explicit acceptance requirements for init-time architecture derivation, unsupported-variant failure behavior, override validation, and validation/perf evidence.

**Plan Changes:**
- Added runtime-state persistence for resolved architecture params.
- Added Python-side override validation before native init.
- Added tests for invalid override handling and init-only architecture derivation evidence.

### Impact Assessment

- Tasks affected: 2.3, 2.4, 2.5, 2.6, 2.7, 4.2, 4.3, 4.4, 4.5, 5.3, 5.4, 5.5, 5.6
- Timeline impact: Low to moderate; sharper acceptance criteria should reduce downstream debugging.
- Dependencies updated: None
