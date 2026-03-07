# Architecture Decisions

## ADR-0001: `cpu` is the default runtime backend

- Date: 2026-03-05
- Status: Accepted
- Flow: `gpu-backend-abstraction_20260304`

### Context

`mogemma` now exposes backend selection semantics (`cpu`, `gpu`, and `gpu:<index>` selectors), but GPU execution is not fully implemented yet. We also need to preserve the current Python/Mojo execution boundary and avoid introducing MAX dependencies in the primary runtime path.

### Decision

Use `cpu` as the initial default backend.

### Rationale

- Preserves compatibility with the existing compiled `mogemma._core` runtime path.
- Keeps behavior deterministic while backend abstraction is rolled out in staged chapters.
- Avoids license/process and distribution implications from MAX in the primary path.
- Provides a low-risk seam for future `gpu` implementation without API churn.

### Consequences

- `cpu` and empty device values resolve to `cpu`.
- `gpu`/`gpu:<index>` selectors normalize to `gpu` identifiers but are not executable until GPU backend implementation lands.
- Follow-up chapters must implement runtime/device semantics and kernel execution for `gpu`.
