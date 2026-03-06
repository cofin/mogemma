# Master PRD: hatch-mojo Upstream Alignment (v0.1.5)

## Context

mogemma currently carries workaround-oriented CI/build configuration that was introduced during Feb 25-26, 2026 while hatch-mojo limitations were still being worked through. The present `pyproject.toml` still includes workaround comments and behavior tied to `cofin/hatch-mojo#7` assumptions:

- `hatch-mojo>=0.1.4` (not explicitly aligned to latest `v0.1.5`)
- Linux `before-build` copy to `/tmp/modular-lib` with `MODULAR_LIB_DIR`
- Linux `sed` toggle to force `bundle-libs = true`
- Linux custom retag `repair-wheel-command` using `wheel tags`
- macOS `repair-wheel-command = ""`
- Default `[tool.hatch.build.targets.wheel.hooks.mojo] bundle-libs = false`

Upstream `../hatch-mojo` is now at `v0.1.5` (`d61bafe`, 2026-02-27), with runtime fixes and updated docs indicating bundled runtime libs work on Linux and macOS (`install_name_tool` + reference rewriting path included).

## North Star

mogemma packaging and CI should be explicitly aligned with current hatch-mojo behavior: keep only workarounds that are still necessary, remove stale overrides, and document each decision with upstream evidence.

## Roadmap

### Chapter 1: `hatch-mojo-upstream-parity_20260301`
Build an evidence-backed parity audit across CI, code comments, docs, and git history (mogemma + hatch-mojo), producing a keep/remove decision matrix for each active workaround.

### Chapter 2: `hatch-mojo-ci-realignment_20260301`
Apply the Chapter 1 decisions to build configuration and CI workflows so Linux/macOS wheel behavior tracks upstream recommendations while preserving release reliability.

### Chapter 3: `hatch-mojo-docs-knowledge-sync_20260301`
Synchronize runbooks and `.agent` knowledge artifacts with the new baseline, including explicit rationale for any retained exceptions.

## Global Constraints

- Planning-first workflow: this PRD defines scope and execution order; implementation occurs in `/flow:implement`.
- Keep release safety as a hard requirement; avoid speculative cleanup without evidence.
- Decisions must cite concrete sources from:
  - `mogemma` (`pyproject.toml`, workflow files, test assertions, docs, git history)
  - `hatch-mojo` (`README.md`, `hatch_mojo/runtime.py`, release commits/tags)
- Any retained override must include a reason and exit criteria.

