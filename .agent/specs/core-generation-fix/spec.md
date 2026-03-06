# Flow: core-generation-fix

## Specification
Current behavior indicates two different quality failures:
1. `gemma3n-e2b-it` emits incoherent/gibberish outputs.
2. Standard-model semantic checks must remain deterministic and fail-fast.

Research confirms the nano issue is not corrupted weights; it is architecture drift in Mojo runtime semantics versus Gemma 3n reference behavior (AltUp, per-layer inputs, and attention/cache behavior).

### Code Analysis Summary
- `src/mo/mogemma/layers.mojo`: nano path currently diverges from reference Gemma3n semantics.
  - AltUp math differs from reference `predict`/`correct` routing behavior.
  - Per-layer input path is incomplete (global projection/norm tensors are loaded but not meaningfully applied in the layer flow).
  - Per-layer embedding lookup uses hardcoded layer count (`30`) instead of runtime-derived axis size.
- `src/mo/mogemma/core.mojo`: expensive runtime pattern.
  - `_build_model` / `_build_nano_model` reconstruct full model descriptors from Python metadata in `step_mojo` and embedding hot paths.
  - This repeatedly re-parses metadata and re-materializes lists on token steps.
- `src/py/mogemma/convert.py`: tensor mapping/invariants are now stronger, but layout validity alone does not ensure semantic parity.
- `tools/validate.py`: deterministic standard factual gate is in place; nano remains failing semantically.

### Primary Files / Evidence Targets
- `src/mo/mogemma/layers.mojo`: nano attention, AltUp, per-layer input, sparsity, and KV-sharing behavior.
- `src/mo/mogemma/core.mojo`: runtime/session descriptor construction, model-slot caching, and hot-path `step_mojo` / embedding entrypoints.
- `src/mo/tests/test_altup_contract.mojo` and `src/mo/tests/test_nano_layers.mojo`: intermediate parity checkpoints for nano-only math.
- `src/py/tests/test_mojo_core.py` and `src/py/tests/test_validate_script.py`: Python-visible regression surface and deterministic validation gates.
- `tools/validate.py` and `tools/benchmark.py`: end-to-end semantic checks and before/after hot-path evidence capture.

### Requirements
1. Nano generation (`gemma3n-e2b-it`) must pass non-gibberish sanity prompts in end-to-end validation.
2. Standard generation (`gemma3-270m-it`) must satisfy deterministic semantic gates (including France -> Paris).
3. Validation and runtime behavior must remain strict fail-fast with explicit errors (no soft fallbacks).
4. Nano runtime must align to reference Gemma3n semantics for:
   - AltUp `predict`/`correct` behavior,
   - per-layer input projection + gate path,
   - attention/cache behaviors relevant to Gemma3n (including KV-sharing/v-norm where required).
5. Mojo runtime must avoid per-token model descriptor reconstruction; parameter descriptors/config should be prepared once at init and reused in generation/embedding loops.
6. Embedding model-id catalog cleanup remains deferred to `embedding-model-id-policy`.
7. Descriptor caching must be measurable: a steady-state generation or embedding run may build the active model descriptor at most once per explicit load/reset cycle, and tests must fail if hot-path rebuild counts increase.
8. The fix must preserve the existing public Python model APIs and continue to fail fast on unsupported runtime states; no silent CPU fallbacks, no hidden variant coercion, and no per-token Python object conversion in the hot path.
9. Validation evidence for the implementation PR must include deterministic intermediate parity checkpoints for nano math, deterministic standard semantic prompts, deterministic nano smoke prompts that detect gibberish regressions, and before/after hot-path evidence (descriptor-build counts plus fixed-prompt latency or tokens/sec).

## Implementation Plan

### Phase 1: Gemma3n Semantic Parity (Correctness First)
- [~] 1.1 `mogemma-yj1.2.1` Replace current nano forward behavior with reference-faithful Gemma3n semantics — nano code paths exist (forward_nano_layer, AltUp, activation sparsity) but semantic parity unverified at runtime.
- [ ] 1.2 `mogemma-yj1.2.5` Add deterministic intermediate-parity checks (selected layer/tensor checkpoints) to prevent regressions while refactoring nano math.
- [ ] 1.3 Freeze the checkpoint set before implementation: AltUp predict/correct outputs, per-layer-map slices, KV-share layer outputs, and final logits for one fixed prompt/token step.

### Phase 2: Runtime Parameter Storage and Config Performance
- [~] 2.1 `mogemma-yj1.2.6` Introduce a one-time model runtime descriptor — runtime dicts built but per-token model reconstruction still occurs.
- [ ] 2.2 `mogemma-yj1.2.7` Remove repeated `_build_*` calls from hot paths and route `step_mojo` / `generate_embeddings_mojo` through cached descriptors.
- [ ] 2.3 Cache runtime-owned layer views, RoPE tables, and variant constants alongside the descriptor so token loops only dereference prepared state.
- [x] 2.3 `mogemma-yj1.2.4` Keep nano conversion/layout invariants as a required guardrail for descriptor initialization.

### Phase 3: Integrated Multi-Model Validation
- [ ] 3.1 `mogemma-yj1.2.3` Execute generation validation for `gemma3-270m-it` and `gemma3n-e2b-it`, capturing semantic pass/fail outcomes.
- [ ] 3.2 `mogemma-yj1.2.8` Add lightweight perf checks proving descriptor-caching removes repeated model-build overhead in token loops.
- [ ] 3.3 Document failure signatures, fixed behavior, and perf deltas in flow learnings after implementation.
- [ ] 3.4 Manual verification: run a fixed 32-token standard prompt and a fixed nano smoke prompt from a cold init and from a warm session, confirming semantic parity, descriptor-build count stability, and no warm-run regression.
