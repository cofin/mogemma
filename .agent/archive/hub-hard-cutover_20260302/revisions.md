## [2026-03-06 00:31] Revision 1

**Type:** both
**Reason:** The cutover plan needed explicit teardown and hot-path rules so the obstore removal PR can be executed without guessing about cleanup or per-file native setup costs.
**Changes:** Added a stricter performance guardrail for downloader lifecycle and a TDD checklist covering obstore-removal assertions, path-safety failures, cleanup semantics, and full-model resolution regression coverage.
