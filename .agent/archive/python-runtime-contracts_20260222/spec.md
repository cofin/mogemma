# Flow: python-runtime-contracts_20260222

## Specification

### Goal

Harden Python runtime wrappers so sync/async generation and embedding behavior is deterministic, type-safe, and consistent with finalized Mojo core contracts.

### Code Analysis Summary

1. `src/py/mogemma/model.py:44` contains custom sampling logic with multiple numeric edge branches and needs contract-backed coverage.
2. `src/py/mogemma/model.py:111` and `src/py/mogemma/model.py:188` perform lazy tokenizer initialization and require deterministic dependency-failure behavior.
3. `src/py/mogemma/model.py:211` and `src/py/mogemma/model.py:124` raise runtime errors for missing core; this error taxonomy should be standardized across code paths.
4. `src/py/mogemma/model.py:254` async streaming uses `asyncio.to_thread` and needs cancellation/termination guarantees.
5. Current tests in `src/py/tests/test_gemma_model.py`, `src/py/tests/test_async.py`, and `src/py/tests/test_embeddings.py` cover key paths but do not comprehensively lock edge contracts.

### Requirements

1. Sync generation behavior must be deterministic for zero-temperature paths and robust for numeric edge conditions.
2. Async wrappers must not leak threads or hang on generator termination.
3. Embedding paths must enforce explicit input shape and output row mapping behavior.
4. Error messages must be consistent and actionable for missing tokenizer/core/runtime failures.
5. Type hints and runtime checks must align with final core return contracts.

### Non-goals

1. Replacing `tokenizers` with a different tokenizer stack.
2. Introducing new public API classes beyond existing release scope.
3. Major refactor of tracing architecture.

## Implementation Plan

### Phase 1: Contract Matrix for Runtime APIs

- [x] 1.1 Document explicit contracts for `EmbeddingModel.embed`, `EmbeddingModel.embed_tokens`, `SyncGemmaModel.generate`, `SyncGemmaModel.generate_stream`, and `AsyncGemmaModel` methods.
- [x] 1.2 Define edge-case expectations for empty prompts, empty logits, `temperature=0`, `top_k=0`, and `top_p` boundaries.
- [x] 1.3 Define dependency-failure expectations for missing tokenizer and missing core extension.
- [x] 1.4 Define async cancellation and stream termination expectations.

### Phase 2: Sync Generation Hardening

- [x] 2.1 Audit `_sample_next_token` numeric stability paths and enforce deterministic behavior for zero-temperature.
- [x] 2.2 Add explicit checks/messages for malformed logits from core.
- [x] 2.3 Validate EOS fallback behavior and token decode assumptions.
- [x] 2.4 Ensure no silent fallback behavior in generation path.
- [x] 2.5 Update tests to lock each branch of sampling logic.

### Phase 3: Async Runtime Hardening

- [x] 3.1 Validate thread handoff behavior for `AsyncGemmaModel.generate`.
- [x] 3.2 Validate async stream termination and `StopIteration` handling.
- [x] 3.3 Add tests for cancellation and prompt edge cases.
- [x] 3.4 Ensure async wrappers propagate meaningful exceptions from sync path.

### Phase 4: Embedding Path Hardening

- [x] 4.1 Tighten input validation for `embed_tokens` and row-count consistency checks.
- [x] 4.2 Add tests for malformed shapes/dtypes and expected failures.
- [x] 4.3 Ensure `embed` and `embed_tokens` contract behavior is explicit in docstrings.
- [x] 4.4 Ensure tokenizer requirement messaging remains precise and concise.

### Phase 5: Error Taxonomy and Type Alignment

- [x] 5.1 Normalize `ModuleNotFoundError` and `RuntimeError` messages across model/runtime paths.
- [x] 5.2 Align type hints with finalized core contracts and facade exports.
- [x] 5.3 Remove ambiguous Any-like paths where strict typing is feasible.
- [x] 5.4 Ensure pyright/mypy constraints remain green under strict settings.

### Phase 6: Validation and Evidence

- [x] 6.1 Run targeted runtime test suites and verify deterministic outcomes.
- [x] 6.2 Run full Python tests to confirm no regressions.
- [x] 6.3 Record test evidence and contract confirmations in Beads task notes.

## Test Plan

1. `uv run pytest src/py/tests/test_gemma_model.py -q`
2. `uv run pytest src/py/tests/test_async.py -q`
3. `uv run pytest src/py/tests/test_embeddings.py -q`
4. `uv run pytest src/py/tests -q`
5. `make test`

## Validation Criteria

1. Runtime API contract matrix is implemented and test-locked.
2. Sync/async behavior is deterministic and non-hanging across edge cases.
3. Error taxonomy is consistent and concise across failure paths.
4. Full Python test suite passes with no new regressions.

## Risks and Mitigations

1. Risk: Sampling changes alter output expectations.
   Mitigation: Use explicit deterministic tests for zero-temperature and branch tests for stochastic paths.
2. Risk: Async cancellation semantics are environment-sensitive.
   Mitigation: Use focused tests with timeout and explicit termination assertions.
