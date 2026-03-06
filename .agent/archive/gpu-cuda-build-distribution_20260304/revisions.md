## [2026-03-06 00:27] Revision 1

**Type:** both
**Reason:** Tighten packaging/release planning so the CUDA distribution PR can land in one pass without policy gaps, hidden CI expansion, or hot-path validation mistakes.
**Changes:** Added explicit artifact manifest requirements, bounded PR-vs-release matrix rules, concrete file/test touch points, cached capability-check constraints, and more specific CI/release evidence tasks.
