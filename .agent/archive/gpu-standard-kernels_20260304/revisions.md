## [2026-03-06 00:28] Revision 1

**Type:** both
**Reason:** Tighten the chapter so a single-pass implementation PR can land without ambiguity and without accidental hot-path regressions.
**Changes:** Added explicit module ownership (`core.mojo`, `layers.mojo`, `ops.mojo`/GPU sibling), non-goals, hot-path invariants, session-level fallback rules, concrete validation surfaces, and sharper TDD sequencing focused on parity, fallback latching, and regression coverage.
