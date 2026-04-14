# Learnings - Launch Cleanup

## Execution Learnings

- **Scope Containment:** Keeping cleanup work in one flow and one Beads epic (`mogemma-gih`) reduced fragmentation and made rollback decisions straightforward.
- **Fail-Fast over Backward Compatibility:** Replacing placeholder generation paths with hard errors improved reliability and debugging clarity in downstream callers.
- **API Rename Safety:** Renaming `GemmaModel` → `SyncGemmaModel` must be treated as a coordinated change across public exports, tests, and docs to avoid latent import failures.
- **Dead-Code Retirement Pattern:** Removing unused native exports (`generate_text_mojo`, `process_image_mojo`) and Python wrappers (`VisionGemmaModel`) together with associated tests prevented broken-import surprises.
- **Dependency Hygiene:** Moving Mojo tooling toward build-time boundaries avoided heavy runtime requirements and simplified installation.

## Practical Notes

- Legacy or partially implemented modules should be removed fully, including tests and benchmark references, not partially shimmed.
- Placeholder files in toolchains (`scripts/benchmark.py` references) should be verified during cleanup to avoid dead references in CI and docs.
