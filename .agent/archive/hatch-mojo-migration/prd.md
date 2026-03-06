# Master PRD: hatch-mojo Migration

## Context

mogemma has been using a custom build hook (`tools/hatch_build.py`) to compile Mojo sources into a Python extension (`_core.so`). The `hatch-mojo` plugin was extracted as a standalone library to make this reusable. The `feat/cleanup` branch has already:

- Added `hatch-mojo` to `[build-system] requires`
- Replaced the custom hook config with declarative `[[tool.hatch.build.targets.wheel.hooks.mojo.jobs]]`
- Updated the Makefile build target from direct `mojo build` to `uv build --wheel`
- Deleted `tools/hatch_build.py`
- Deleted stale `.agent/code-styleguides/` files

**What remains:** fix broken references to the deleted file, update project documentation, and verify the build pipeline works.

## North Star

mogemma builds cleanly with `hatch-mojo` as its only Mojo compilation mechanism — no custom hooks, no dead code, all tests pass.

## Roadmap

### Chapter 1: `cleanup-and-migrate`

Complete the migration by fixing broken tests, updating internal docs, and verifying the build.

## Global Constraints

- No new features — this is cleanup only
- The `hatch-mojo` library itself is NOT modified here (issues tracked in `hatch-mojo/fix-me.md`)
- All changes should be on `feat/cleanup` branch
