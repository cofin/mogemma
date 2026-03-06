# Flow: quality-gates-zero-warning_20260222

## Specification

### Goal

Make quality gates strict and release-enforceable by ensuring `make lint` and CI fail on any lint/type issue, while minimizing suppressions to only justified cases.

### Code Analysis Summary

1. `Makefile` lint target currently includes soft-fail patterns for mypy/pyright (`|| echo`), which allows false-green local checks.
2. `pyproject.toml` enforces strict mypy and broad ruff rule selection (`select = ["ALL"]`) with targeted ignore lists.
3. Existing suppressions in source/tests are partly intentional but require explicit audit and justification.
4. CI workflows already run separate lint/mypy/pyright jobs and can be aligned with strict local semantics.

### Requirements

1. `make lint` must fail fast on any `ruff`, `mypy`, or `pyright` issue.
2. Suppression surface must be minimized and documented.
3. CI quality jobs must mirror local strictness.
4. Release gates must include explicit quality command set with reproducible outputs.
5. No broad suppression additions to silence unresolved defects.

### Non-goals

1. Lowering strictness to satisfy short-term convenience.
2. Adopting an alternate linter/type-checker stack for this release.
3. Reformatting unrelated files without rule-driven reason.

## Implementation Plan

### Phase 1: Enforce Strict Local Gates

- [x] 1.1 Remove soft-fail `|| echo` behavior from `Makefile` lint/type commands.
- [x] 1.2 Ensure `make lint` exits non-zero on first quality error.
- [x] 1.3 Add or update dedicated `make check-release` target that runs strict quality + tests in release order.
- [x] 1.4 Verify local command behavior in clean environment.

### Phase 2: Suppression Audit and Reduction

- [x] 2.1 Inventory all `ruff noqa`, `pyright ignore`, and `mypy ignore` usages in `src/py/mogemma` and tests.
- [x] 2.2 Remove suppressions that are no longer required.
- [x] 2.3 For remaining suppressions, add concise justification comments where appropriate.
- [x] 2.4 Ensure `pyproject.toml` ignore lists are minimal and explicit.

### Phase 3: Rule-Conformant Fixes

- [x] 3.1 Fix code issues discovered when suppressions are reduced.
- [x] 3.2 Fix test issues without reducing assertion strength.
- [x] 3.3 Ensure typing facade pattern remains clean under strict analysis.
- [x] 3.4 Keep changes scoped to quality compliance, no unrelated refactor.

### Phase 4: CI Gate Alignment

- [x] 4.1 Align `.github/workflows/ci.yml` command semantics with strict local `make lint` expectations.
- [x] 4.2 Ensure CI fails on any quality issue with clear job logs.
- [x] 4.3 Add/adjust workflow checks for release branch or release events if required.
- [x] 4.4 Validate CI command parity with documented local runbook.

### Phase 5: Validation and Evidence

- [x] 5.1 Run `make lint` and capture clean output evidence.
- [x] 5.2 Run `make test` and ensure no regressions from quality fixes.
- [x] 5.3 Record suppression delta and rationale in flow notes.
- [x] 5.4 Mark chapter complete only with zero-warning proof artifacts.

## Test Plan

1. `make lint`
2. `uv run ruff check src/py`
3. `uv run mypy src/py/mogemma`
4. `uv run pyright src/py/mogemma`
5. `make test`

## Validation Criteria

1. `make lint` hard-fails on any quality issue.
2. Local and CI quality semantics are aligned.
3. Suppressions are reduced to justified minimum.
4. Full test suite remains green.

## Risks and Mitigations

1. Risk: Strict gating exposes large backlog of issues.
   Mitigation: Fix by severity and keep chapter bounded to release-critical scope.
2. Risk: Over-aggressive suppression removal breaks developer ergonomics.
   Mitigation: Keep justified suppressions with explicit rationale, avoid blanket removals.
