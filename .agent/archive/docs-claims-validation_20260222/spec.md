# Flow: docs-claims-validation_20260222

## Specification

### Goal

Ensure every README/docstring/public claim is accurate, concise, and evidence-backed; remove marketing language and keep documentation aligned with tested behavior.

### Code Analysis Summary

1. `README.md` contains feature claims and quick-start examples that must map directly to tested behavior.
2. Runtime behavior spans `src/py/mogemma/model.py`, `src/py/mogemma/hub.py`, and optional dependency facades in `src/py/mogemma/_typing.py` and `src/py/mogemma/typing.py`.
3. Existing tests in `src/py/tests` provide claim evidence for major paths but require explicit mapping to docs.
4. Prior user-requested direction requires concise language with no marketing-style performance claims.

### Requirements

1. All feature claims must be explicitly validated against implementation and tests.
2. Marketing language and vague performance claims must be removed or rewritten to measurable technical wording.
3. Examples must be executable and aligned with current API signatures.
4. Optional dependency behavior must be documented accurately after model-delivery policy changes.
5. Documentation updates must remain concise and release-focused.

### Non-goals

1. Building a full documentation site framework in this flow.
2. Adding long-form conceptual tutorials not required for first release.
3. Introducing speculative future-roadmap claims.

## Implementation Plan

### Phase 1: Claims Inventory and Evidence Map

- [x] 1.1 Create a claims inventory from README and public docstrings.
- [x] 1.2 Map each claim to supporting code paths and tests.
- [x] 1.3 Flag unsupported or partially supported claims for rewrite/removal.
- [x] 1.4 Add unresolved-claim list and required implementation dependencies.

### Phase 2: Language and Content Revision

- [x] 2.1 Rewrite feature section wording to concise technical statements.
- [x] 2.2 Remove vague superlatives and marketing-style wording.
- [x] 2.3 Ensure install/dependency sections reflect final release dependency policy.
- [x] 2.4 Update troubleshooting language for missing model/dependency/runtime scenarios.

### Phase 3: Example Verification

- [x] 3.1 Verify embedding examples run with release API paths.
- [x] 3.2 Verify sync generation examples run with release API paths.
- [x] 3.3 Verify async streaming examples run and terminate correctly.
- [x] 3.4 Add or update tests that execute README-critical examples.

### Phase 4: Public API Docstring Alignment

- [x] 4.1 Audit docstrings in `src/py/mogemma` for claim and behavior accuracy.
- [x] 4.2 Tighten docstrings to concise, concrete language.
- [x] 4.3 Ensure exceptions and constraints are documented where behavior is non-obvious.
- [x] 4.4 Confirm consistency between module-level docs and README.

### Phase 5: Final Documentation Gate

- [x] 5.1 Add a docs claim-validation checklist to release evidence.
- [x] 5.2 Re-run lint/tests relevant to docs examples.
- [x] 5.3 Record evidence mapping from claims to tests in Beads notes.

## Test Plan

1. `uv run pytest src/py/tests -q`
2. Any example-validation test commands added in this flow
3. `make lint`

## Validation Criteria

1. Every claim in README/public docstrings has implementation + test evidence.
2. Marketing-style language is removed in favor of concise technical wording.
3. Examples run against real API behavior.
4. Dependency and troubleshooting guidance is accurate for release.

## Risks and Mitigations

1. Risk: Docs drift after late implementation changes.
   Mitigation: Keep this chapter late in sequence and require final revalidation.
2. Risk: Over-condensed wording removes necessary guidance.
   Mitigation: Keep concise but include explicit constraints and failure guidance.
