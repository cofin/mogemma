# Flow: Gemma Softcapping

> **Status**: [~] In Progress
> **Epic**: [mogemma-softcapping](mogemma-softcapping)

## 1. Background
Gemma 4 models use logit softcapping (`cap * tanh(value / cap)`) during training to stabilize attention scores and prevent logit explosion. Without this in the inference runtime, the model produces "gibberish" or repetitive output because the attention scores and final logits fall outside the training range.

## 2. Specification
- **Final Logit Softcapping**: Applied to the output of the LM head. Default: `30.0`.
- **Attention Logit Softcapping**: Applied to query-key dot products before scaling/masking/softmax. Default: `50.0`.
- **Formula**: `capped = cap * tanh(value / cap)`.
- **No-op**: If `cap <= 0.0`, the operation is bypassed.

## 3. Implementation Plan

### Phase 1: Metadata & Parser (Python)
- [ ] **Task: Parser Preservation** (`mogemma-softcapping.1`)
  - Update `_parse_gemma4_architecture` in `src/py/mogemma/model.py`.
  - Add tests in `src/py/tests/test_gemma4_config_wiring.py`.
- [ ] **Task: Conversion Metadata** (`mogemma-softcapping.2`)
  - Update `_VARIANT_CONSTANTS` in `src/py/mogemma/convert.py`.
  - Add tests in `src/py/tests/test_convert.py`.

### Phase 2: Backend Ops (Mojo)
- [ ] **Task: CPU Softcap Primitive** (`mogemma-softcapping.3`)
  - Implement `softcap` in `src/mo/mogemma/ops.mojo`.
  - Add `ComputeBackend.softcap` method.
  - Add tests in `src/mo/tests/test_ops.mojo`.
- [ ] **Task: GPU Softcap Kernel** (`mogemma-softcapping.4`)
  - Implement `softcap_kernel` and `GPUBackend.softcap` in `src/mo/mogemma/ops_gpu.mojo`.

### Phase 3: Forward Pass Wiring (Mojo)
- [ ] **Task: Attention Softcap Application** (`mogemma-softcapping.5`)
  - Thread `attn_logit_softcapping` through `layers.mojo`.
  - Apply before softmax in `forward_sliding_attention`, `forward_full_attention`, and `forward_moe_attention`.
  - Create `src/mo/tests/test_softcap.mojo`.
- [ ] **Task: Final Logit Softcap Application** (`mogemma-softcapping.6`)
  - Apply after LM-head projection in `forward_gemma4_step`, `forward_gemma4_ple_step`, and `forward_gemma4_moe_step`.
- [ ] **Task: Runtime Entrypoints** (`mogemma-softcapping.7`)
  - Wire values from `architecture_overrides` in `src/mo/mogemma/core.mojo`.
  - Pass through `_run_step` and `step_mojo`.

### Phase 4: Verification & Cleanup
- [ ] **Task: Regression Testing** (`mogemma-softcapping.8`)
  - Full `make check-all`.
  - Real-weight smoke test with `google/gemma-4-E2B-it`.
- [ ] **Task: Legacy Plan Removal** (`mogemma-softcapping.9`)
  - Remove `docs/superpowers/plans/2026-04-25-gemma-softcapping.md`.

## 4. Learnings & Patterns
- Softcap constants are family-wide facts for Gemma 4 variants.
- Implementation must be backend-aligned (CPU/GPU) to maintain parity.
