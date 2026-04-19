# Flow: Fix Gemma 4 Gibberish Generation (Logits Soft-Capping)

## Tasks

- [x] **Update `convert.py` to Emit Soft-Cap**
    -   File: `src/py/mogemma/convert.py`
    -   Action: Update `_VARIANT_CONSTANTS` so that `"final_logit_softcapping": 30.0` is included for all Gemma 4 variants (`base`, `ple`, `moe`).
    -   Test: Update or add a test in `src/py/tests/test_convert.py` to ensure `final_logit_softcapping` is synthesized into the `config.json`.

- [x] **Parse Soft-Cap in Python Runtime**
    -   File: `src/py/mogemma/model.py`
    -   Action: Read `final_logit_softcapping` from the parsed `config.json` (defaulting to `30.0` if not present) and add it to the initialized `llm` dictionary passed to the Mojo core.
    -   Test: Ensure the test suite still passes.

- [~] **Implement Logit Capping in Mojo Core**
    -   Files: `src/mo/mogemma/core.mojo`, `src/mo/mogemma/layers.mojo` (or wherever final logits are computed)
    -   Action: Immediately after calculating the final logits, check if `final_logit_softcapping` > 0.0. If so, iterate over the `vocab_size` elements in the `out_logits_ptr` buffer and apply `val = tanh(val / soft_cap) * soft_cap`.
    -   Action: For GPU, ensure this is handled efficiently.
    -   Test: `tools/validate.py --mode llm` should now output coherent answers instead of gibberish. (Validation still failing with semantic error)

## Phases
-   **Phase 1**: Python implementation (Tasks 1 & 2)
-   **Phase 2**: Mojo implementation (Task 3) and Verification