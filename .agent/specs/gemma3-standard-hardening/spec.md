# Flow: gemma3-standard-hardening

## Specification

### Code Analysis Summary

**Problem:** Three hardcoded assumptions in `core.mojo` make the inference engine brittle:

1. **`head_dim = 256`** (lines 99, 158, 222) — Used to compute `num_heads` and `num_kv_heads` from projection weight shapes. If any Gemma variant uses a different head dimension, all attention math breaks silently.

2. **`embedding_dim = 768`** check (line 28) — `_ensure_embedding_matrix` rejects any model whose hidden_size != 768, even though gemma-3-1b-it has hidden_size=1536 for embeddings.

3. **`GenerationConfig.model_path = "gemma3n-e2b-it"`** (`config.py:51`) — Points to Gemma 3 Nano which the Mojo backend cannot load (different architecture).

**Solution approach (derive with override):**

The safetensors files already contain `q_norm.weight` and `k_norm.weight` per layer (written by `convert.py`). These are 1D tensors with shape `(head_dim,)`. By loading them into `LayerWeights` and reading their shape, Mojo can derive `head_dim` directly — no guessing, no config file needed.

For the override path: modify `init_model_mojo` to accept an optional architecture params dict from Python. If the caller provides `head_dim`, `rope_base`, etc., those take precedence over derived values.

### Files Affected

| File | Change Type | Description |
|------|-------------|-------------|
| `src/mo/mogemma/model.mojo` | Edit | Add `q_norm` and `k_norm` TensorInfo fields to `LayerWeights` |
| `src/mo/mogemma/core.mojo` | Edit | Load q_norm/k_norm in `_build_model`; derive `head_dim` from `q_norm.shape_0`; remove hardcoded 256 and 768; accept optional arch params |
| `src/mo/mogemma/layers.mojo` | Edit | Add QK norm application in `forward_attention` (currently skipped — these norms exist in the weights but aren't used) |
| `src/py/mogemma/config.py` | Edit | Change `GenerationConfig.model_path` default to `"gemma3-270m-it"` |
| `src/py/mogemma/model.py` | Edit | Pass arch override dict through `_initialize_llm` → `_core.init_model` |
| `src/mo/tests/test_layers.mojo` | Edit | Update test fixtures for new QK norm weights in LayerWeights |
| `src/py/tests/test_gemma_model.py` | Edit | Update tests for new default model path |

### Key Design Decisions

1. **QK norms as head_dim source:** `q_norm.weight` shape is `(head_dim,)` — this is the most reliable way to derive head_dim without external config files. We already save these tensors in `convert.py`.

2. **Embedding dim from tensor shape:** Replace `if Int(py=embeddings.shape[1]) != 768` with a dynamic check against `hidden_size` derived from `model.embed_tokens.shape_1`.

3. **Override mechanism:** Python can optionally pass `{"head_dim": N, "rope_base": M, ...}` to `init_model`. This prepares the system for Nano (Chapter 2+3) where architecture detection from weights alone may not suffice.

4. **QK norm in attention:** While fixing dimensions, also wire in the QK norm application. Gemma 3 uses per-head QK normalization that the current `forward_attention` doesn't apply. This is a correctness fix for output quality.

---

## Acceptance Requirements

1. Architecture-derived values (`head_dim`, `hidden_size`, norm presence, supported variant) must be resolved once during init and stored in runtime state; `step_mojo` and embedding hot paths must not rediscover shape metadata per call.
2. Unsupported or ambiguous model variants must fail at init with deterministic, user-visible errors before any token step occurs.
3. `architecture_overrides` must be validated in Python config objects before `_core.init_model` is called; unknown keys, invalid types, and unsupported override combinations must fail fast.
4. Standard-model loading must accept any embedding width consistent with tensor metadata rather than a hardcoded `768`, while nano checkpoints remain explicitly gated to the nano flow until their runtime path is implemented.
5. The implementation PR must include deterministic validation evidence for standard-model generation and embedding smoke tests, variant detection/rejection behavior, override pass-through behavior, and no-regression hot-path evidence showing architecture handling adds no repeated per-token work.

## Implementation Plan

### Phase 1: Mojo Model Extension (src/mo/mogemma/model.mojo)

- [x] 1.1 Add `q_norm: TensorInfo` and `k_norm: TensorInfo` fields to `LayerWeights` struct
- [x] 1.2 Update `LayerWeights.__init__` to zero-initialize q_norm and k_norm
- [x] 1.3 Add `pre_feedforward_layernorm: TensorInfo` and `post_feedforward_layernorm: TensorInfo` to `LayerWeights` (these exist in safetensors but aren't loaded — needed for correctness on models that use 4 norms)

### Phase 2: Dynamic Dimension Derivation (src/mo/mogemma/core.mojo)

- [x] 2.1 Load q_norm, k_norm, pre/post_feedforward_layernorm in `_build_model` for each layer
- [x] 2.2 In `init_model_mojo`: derive `head_dim` from `model_weights.layers[0].q_norm.shape_0` (with fallback to 256 if q_norm has zero shape — backward compat with older cached models)
- [~] 2.3 Remove hardcoded `var head_dim = 256` from `step_mojo` — q_norm derivation added, but 256 fallback remains
- [~] 2.4 Remove hardcoded `var head_dim = 256` from `generate_embeddings_mojo` — q_norm derivation added, but 256 fallback remains
- [ ] 2.5 Replace `_ensure_embedding_matrix` check `!= 768` with `!= hidden_size` derived from `model.embed_tokens.shape_1`
- [ ] 2.6 Add optional arch params dict to `init_model_mojo` signature; if `head_dim` key present, use override value
- [ ] 2.7 Persist resolved architecture params in runtime state so token/embedding entrypoints consume prepared values instead of re-reading tensor metadata.

### Phase 3: QK Norm in Attention (src/mo/mogemma/layers.mojo)

- [x] 3.1 Add `q_norm` and `k_norm` parameters to `forward_attention` signature
- [x] 3.2 After Q projection, apply RMS norm per-head using q_norm weights
- [x] 3.3 After K projection, apply RMS norm per-head using k_norm weights
- [x] 3.4 Update `forward_layer` to pass q_norm/k_norm to `forward_attention`
- [x] 3.5 Update `forward_step` and `forward_sequence` to propagate the new parameter

### Phase 4: Python Config & Override (config.py, model.py)

- [x] 4.1 Change `GenerationConfig.model_path` default from `"gemma3n-e2b-it"` to `"gemma3-270m-it"`
- [ ] 4.2 Add `architecture_overrides: dict[str, int | float] | None = None` field to both config classes
- [ ] 4.3 Update `_initialize_llm` in model.py to pass architecture overrides to `_core.init_model`
- [ ] 4.4 Add `ModelVariant` detection utility that checks tensor names to identify standard vs Nano, raising `ValueError` for unsupported architectures
- [ ] 4.5 Validate override keys/types in Python config construction so unsupported runtime-shaping inputs fail before native init.

### Phase 5: Tests & Validation

- [ ] 5.1 Update `test_layers.mojo` fixtures to include q_norm/k_norm in LayerWeights construction
- [ ] 5.2 Update `test_gemma_model.py` for new default GenerationConfig model path
- [ ] 5.3 Add test: model variant detection rejects Nano with clear error message
- [ ] 5.4 Add test: architecture override dict is passed through to init_model
- [ ] 5.5 Add test: invalid override keys/types fail at config/model construction time with deterministic messages
- [ ] 5.6 Capture a short perf sanity check (or build counter evidence) proving architecture derivation is init-only work
- [ ] 5.7 Run full `uv run pytest src/py/tests/ -x -q` — all tests pass
- [ ] 5.8 Run `uv run ruff check` on all modified files — clean
- [ ] 5.9 Run `make build && make test` for Mojo compilation + Mojo tests (if available)
