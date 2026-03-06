## [2026-03-06 00:31] Revision 1

**Type:** both
**Reason:** Tightened the downloader architecture spec so the implementation PR can land without ambiguity around ABI ownership, compatibility scope, or cache-atomicity guarantees.
**Changes:** Added Mojo/Python compatibility guardrails, an explicit compatibility/contract-verification phase, and a red-green-regression checklist covering ownership, cleanup, and stable error mapping.
