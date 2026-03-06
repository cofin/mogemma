# Flow: validation-enhancements

## Specification
The current validator reports success when generation simply runs without crashing. This chapter adds deterministic semantic gates and explicit failure taxonomy so validation signals model quality regressions reliably.

### Code Analysis Summary
- `tools/validate.py` currently uses a single prompt (`"What is the capital of France?"`) and prints the response without semantic pass/fail checks.
- `tools/validate.py` sets generation parameters to `temperature=0.7` for validation; this is non-deterministic and weak for factual contract testing.
- `tools/validate.py` exits on any exception but does not distinguish:
  - model resolution/download failures,
  - runtime/backend failures,
  - semantic quality failures.
- Existing tests already enforce fail-fast model resolution behavior:
  - `src/py/tests/test_hub.py`
  - `src/py/tests/test_embeddings.py`

### Requirements
1. Define a semantic prompt suite (3-5 prompts) with explicit acceptance criteria.
2. Execute at least one deterministic validation mode (`temperature=0`) for factual contracts.
3. Emit explicit diagnostic categories for:
   - `model_resolution_failure`
   - `runtime_inference_failure`
   - `semantic_validation_failure`
4. Preserve fail-fast behavior: validation exits non-zero at first contract failure.

## Implementation Plan

### Phase 1: Semantic Validation Suite
- [x] 1.1 `mogemma-yj1.3.3` Define 3-5 semantic prompts and acceptance rules, including a factual France/Paris gate.
- [x] 1.2 `mogemma-yj1.3.1` Implement semantic assertion execution in validator with deterministic decoding settings.

### Phase 2: Failure Diagnostics
- [x] 2.1 `mogemma-yj1.3.2` Add explicit failure taxonomy and messages separating resolution/runtime/semantic failure classes.
- [x] 2.2 `mogemma-yj1.3.2` Ensure validator output provides actionable remediation hints per failure class.
