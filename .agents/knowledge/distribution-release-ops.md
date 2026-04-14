# Knowledge: Distribution & Release Ops

> Flow: distribution-release-ops_20260222 | Archived 2026-02-22

## Key Changes

Validated wheel/sdist artifact matrix. Created explicit release runbook. Added isolated-environment smoke tests. Defined go/no-go checklist.

## Patterns

- **Artifact validation:** Test both wheel and sdist in isolated environments; verify imports and smoke paths.
- **Runbook-driven releases:** Codify release steps explicitly; eliminate implicit assumptions.
- **Rollback planning:** Document recovery paths for build and publish failures.

## Gotchas

- Platform-specific build failures close to release are costly — validate matrix early.
- Packaging metadata drift causes install errors only in clean environments.
- Mojo toolchain support varies by platform — keep claimed matrix realistic.
- Wheel includes `_core.so` — verify it exists in build output before publish.
