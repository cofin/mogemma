# Flow: hatch-mojo-upstream-parity_20260301

## Specification

### Goal

Establish whether mogemma’s current hatch-mojo workarounds are still required after upstream updates through `hatch-mojo` `v0.1.5`, and produce a decision matrix that Chapter 2 can implement directly.

### Code Analysis Summary

Files analyzed:

- `pyproject.toml`
- `.github/workflows/ci.yml`
- `.github/workflows/publish.yml`
- `src/py/tests/test_mojo_namespace_layout.py`
- `docs/release-runbook.md`
- `.agent/knowledge/pypi-cibuildwheel.md`
- `.agent/knowledge/cleanup-and-migrate.md`
- `.agent/archive/hatch-mojo-migration/prd.md`
- `.agent/archive/cleanup-and-migrate/spec.md`
- `../hatch-mojo/README.md`
- `../hatch-mojo/hatch_mojo/runtime.py`
- `../hatch-mojo/hatch_mojo/plugin.py`
- `../hatch-mojo/hatch_mojo/config.py`

Git history analyzed:

- `mogemma`: `f32c8b6`, `4dd3a88`, `6e56742`, `6076d81`, `10e6e8c`
- `hatch-mojo`: `f6b1514` (`v0.1.4`), `d61bafe` (`v0.1.5`)

Key findings:

1. `pyproject.toml` currently contains active workaround settings and comments tied to `cofin/hatch-mojo#7` (Linux env copy, Linux sed toggle, macOS empty repair command, `bundle-libs = false` default).
2. Current `build-system` constraint is `hatch-mojo>=0.1.4`; latest upstream in local source is `v0.1.5`.
3. Upstream README now documents `bundle-libs = true` behavior on both Linux and macOS and recommends standard macOS delocate command, not an empty repair override.
4. Upstream runtime now includes platform-aware modular-lib discovery and robust macOS dylib patching/resigning routines, reducing risk behind original workaround comments.
5. Project knowledge notes still include outdated “blocked” framing for prior Linux GLIBCXX constraints and do not reflect the current local workaround stack.

### Requirements

1. Produce a full inventory of active workaround/override behavior with file-level references.
2. Map each workaround to current upstream hatch-mojo behavior and version history evidence.
3. Produce a keep/remove decision matrix with risk notes and fallback path for each item.
4. Output a Chapter 2 “change contract” listing exact config/workflow edits implied by approved decisions.

### Scope

In scope:

- Build/packaging config and CI behavior.
- Test assertions that lock workaround behavior.
- Release/docs/knowledge references that encode workaround assumptions.

Out of scope:

- Direct implementation of config/workflow changes.
- Runtime feature work unrelated to packaging/CI alignment.

## Implementation Plan

### Phase 1: Inventory and provenance

- [x] 1.1 Build a workaround inventory from `pyproject.toml`, CI workflows, tests, and docs (`mogemma-227.1.1`).
- [x] 1.2 Link each workaround to originating `mogemma` commits and rationale text from commit history (`mogemma-227.1.1`).

### Phase 2: Upstream parity map

- [x] 2.1 Extract current upstream recommendations and behavior from `../hatch-mojo/README.md` and runtime/plugin/config code (`mogemma-227.1.2`).
- [x] 2.2 Map `v0.1.4` and `v0.1.5` changes to the specific workaround assumptions tracked in Phase 1 (`mogemma-227.1.2`).

### Phase 3: Decision matrix and contract

- [x] 3.1 Produce keep/remove matrix for each workaround with risk, confidence, and fallback plan (`mogemma-227.1.3`).
- [x] 3.2 Define Chapter 2 implementation contract: exact files, target deltas, and required validation evidence (`mogemma-227.1.4`).
