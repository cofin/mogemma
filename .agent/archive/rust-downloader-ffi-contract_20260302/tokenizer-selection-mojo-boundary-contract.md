# Tokenizer Selection at Mojo Boundary Contract (`mogemma-h4n.2.5`)

## Purpose

Define tokenizer selection ownership and deterministic behavior at the Mojo boundary, removing tokenizer-selection responsibility from Python hub logic.

## Ownership Rules

1. Tokenizer selection source-of-truth is Mojo boundary logic.
2. Python hub layer provides model request context only.
3. Rust downloader performs artifact transfer based on selection results from Mojo boundary.

## Inputs to Tokenizer Selection

1. `model_id`
2. `revision`
3. `artifact_selection` policy
4. Optional explicit tokenizer override (if exposed by API contract)

## Selection Behavior

1. Determine tokenizer artifact candidates from remote manifest metadata.
2. Apply deterministic priority order when multiple tokenizer files exist.
3. Validate chosen tokenizer artifact naming/format constraints.
4. Return a single canonical tokenizer artifact target or explicit no-tokenizer outcome.

## Deterministic Priority Rules

1. Exact canonical tokenizer artifact names are preferred over aliases.
2. Format-compatible tokenizer files are preferred over deprecated variants.
3. If multiple equal-priority candidates remain, choose lexicographically stable winner.

## Failure Behavior

1. If `artifact_selection = model_plus_tokenizer` and no valid tokenizer exists, return terminal `NOT_FOUND`.
2. If tokenizer artifact exists but fails integrity checks, return terminal `INTEGRITY`.
3. No Python-side fallback tokenizer-selection path is allowed in hard cutover mode.

## Boundary Output Contract

1. `tokenizer_path: str | None` (local canonical path after successful materialization)
2. `tokenizer_artifact_name: str | None` (selected remote artifact identity)
3. `tokenizer_required: bool` (derived from selection policy)
4. `tokenizer_status: str` (`resolved`, `not_requested`, or `failed`)

## Chapter 3 Handoff Requirements

1. `HubManager.download()` must treat tokenizer path as boundary output, not recompute selection.
2. Chapter 3 tokenizer/artifact handoff task (`mogemma-h4n.3.4`) must follow this contract exactly.
3. Chapter 4 reliability matrix must include tokenizer-missing and tokenizer-integrity scenarios.
