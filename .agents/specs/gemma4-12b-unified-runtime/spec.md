# Flow: Gemma 4 12B Unified Runtime

**Flow ID:** `gemma4-12b-unified-runtime`  
**Beads Epic:** `mogemma-cpx`  
**Status:** Planned  
**Next Skill:** `flow-execution`

## Beads Tasks

- `mogemma-cpx.1` — Lock Gemma 4 12B nested config contracts.
- `mogemma-cpx.2` — Add CPU variable-head runtime geometry.
- `mogemma-cpx.3` — Enable 12B text-only initialization.
- `mogemma-cpx.4` — Define 12B inventory and loader boundaries.
- `mogemma-cpx.5` — Verify and document 12B text runtime support.

## Specification

### Goal

Implement the next Gemma 4 support slice: make `google/gemma-4-12B-it`
usable for text generation from a local safetensors checkpoint while keeping
image/audio unified multimodal inputs explicitly gated. The completed
`gemma4-working-order-recovery` flow already recognizes 12B and rejects it
truthfully; this flow removes that text-runtime gate only after the 12B config,
runtime geometry, and focused parity contracts are implemented.

### Source Research

Use these already-gathered sources:

- `.agents/research/gemma4-working-order-20260705/research.md`
- Official 12B config refreshed on 2026-07-06 from
  `https://huggingface.co/google/gemma-4-12B-it/raw/main/config.json`

Key 12B config facts to preserve in tests:

```json
{
  "architectures": ["Gemma4UnifiedForConditionalGeneration"],
  "model_type": "gemma4_unified",
  "image_token_id": 258880,
  "audio_token_id": 258881,
  "text_config": {
    "model_type": "gemma4_unified_text",
    "hidden_size": 3840,
    "intermediate_size": 15360,
    "num_hidden_layers": 48,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "num_global_key_value_heads": 1,
    "head_dim": 256,
    "global_head_dim": 512,
    "attention_k_eq_v": true,
    "sliding_window": 1024,
    "max_position_embeddings": 262144,
    "layer_types": ["sliding_attention", "...", "full_attention"],
    "rope_parameters": {
      "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
      "sliding_attention": {"rope_theta": 10000.0}
    }
  }
}
```

`model.safetensors.index.json` is not present at the public HF raw URL checked
on 2026-07-06, so this flow must not assume HF index-based remote downloads.
Support target is local safetensors first; GCS/HF acquisition remains separate.

### Code Analysis Summary

Files examined:

- `src/py/mogemma/model.py:83-150` defines `DENSE_12B_UNIFIED` and currently
  rejects it unconditionally before Mojo initialization.
- `src/py/mogemma/model.py:153-227` parses only top-level Gemma 4 config keys;
  it does not normalize nested `text_config`, `image_token_id`, `audio_token_id`,
  `max_position_embeddings`, `global_head_dim`, or `"full_attention"` layer names.
- `src/py/mogemma/model.py:702-753` rejects `audio=` early, but image handling
  still routes through the side-tower vision encoder. A 12B model must reject
  `images=` until unified image-token support exists.
- `src/py/mogemma/model_support.py:79-95` marks 12B text/image/audio as
  unsupported and unified multimodal as follow-up.
- `src/py/mogemma/convert.py:34-53` models only `"base" | "ple" | "moe"`.
  12B local safetensors can bypass conversion, but Orbax conversion cannot claim
  12B support until a `"unified"` inventory exists.
- `src/mo/mogemma/core.mojo:643-905` infers one global `head_dim` from layer 0
  and passes that to `KVCache`, `RoPETables`, and every attention layer.
- `src/mo/mogemma/model.mojo:407-561` stores one `KVCache.head_dim` and computes
  one `kv_stride = num_kv_heads * head_dim` for all layers.
- `src/mo/mogemma/model.mojo:564-645` stores one RoPE `head_dim`; 12B needs
  sliding head dim 256 and full/global head dim 512.
- `src/mo/mogemma/layers.mojo:220-324` and `835-930` run full attention using
  the single head dim; full-attention 12B layers need a per-layer head dim.
- `src/mo/mogemma/gpu_context.mojo:405-535` mirrors the single-head-dim KV cache
  on GPU. This flow should gate 12B GPU until CPU variable geometry is stable.
- `src/py/tests/test_variant_detection.py:54-65` covers 12B detection with a
  minimal fixture, but not the real nested config values.
- `src/py/tests/test_mojo_core_gemma4_contract.py:52-84` covers one tiny dense
  standard runtime; it does not cover mixed sliding/full layer dimensions.

### Requirements

#### Functional

- Parse official 12B nested config into architecture overrides without losing
  top-level token ids or nested text fields.
- Map 12B layer types `"sliding_attention"` to `0` and `"full_attention"` to
  `1`; continue accepting existing `"sliding"` / `"full"` fixtures.
- Carry both sliding `head_dim` and full/global `head_dim` through Python and
  Mojo runtime initialization.
- Allocate KV cache offsets/strides per layer so full layers can use
  `num_global_key_value_heads * global_head_dim` while sliding layers use
  `num_key_value_heads * head_dim`.
- Build RoPE tables for sliding and full attention with their own head dims and
  full-attention `partial_rotary_factor = 0.25`.
- Permit 12B initialization and text-only generation once the CPU runtime
  contract is green.
- Reject 12B `images=` and `audio=` with precise messages until unified
  multimodal token insertion is implemented.
- Update `model_support.py` so 12B text is supported only after tests pass;
  keep image/audio/unified multimodal as follow-up.

#### Non-Functional

- Do not add `google/gemma-4-12B-it` to `KNOWN_GCS_MODELS` unless a live
  `gs://gemma-data` probe confirms checkpoint objects.
- Do not claim GPU 12B support in this flow. If 12B is requested with GPU,
  raise an explicit unsupported-device error.
- Keep synthetic tests small; no default test may download or load a real 12B
  checkpoint.
- Keep 12B support local-safetensors-first. Remote acquisition is a separate
  flow unless GCS publishes 12B.

#### API

- Existing `GenerationConfig(model_path=...)` remains compatible.
- 12B text users should be able to instantiate `SyncGemmaModel` with a local
  directory containing `config.json`, `tokenizer.model`, and safetensors once
  runtime tensors are available.
- 12B image/audio calls must fail before side-tower hydration or Mojo side-tower
  processing.

#### Risk

- Variable per-layer head dimension affects KV cache offsets, RoPE tables,
  scratch sizing, attention loops, and GPU cache mirrors. Implement CPU first
  and explicitly gate GPU for 12B.
- The real 12B tensor names may differ from existing dense names. This plan
  requires a live local inventory task before any support-status flip.
- Full 262K context can allocate enormous CPU KV buffers. Tests must use small
  `architecture_overrides["max_seq_len"]` values and docs must warn about real
  memory requirements.

## Implementation Plan

### Phase 1: Lock 12B Config Contracts

- [ ] 1.1 Add tests for the official nested 12B config.

  Target: create `src/py/tests/test_gemma4_12b_config.py`.

  Use an inline fixture, matching existing `test_variant_detection.py` style:

  ```python
  GEMMA4_12B_CONFIG = {
      "architectures": ["Gemma4UnifiedForConditionalGeneration"],
      "model_type": "gemma4_unified",
      "image_token_id": 258880,
      "audio_token_id": 258881,
      "text_config": {
          "model_type": "gemma4_unified_text",
          "hidden_size": 3840,
          "intermediate_size": 15360,
          "num_hidden_layers": 48,
          "num_attention_heads": 16,
          "num_key_value_heads": 8,
          "num_global_key_value_heads": 1,
          "head_dim": 256,
          "global_head_dim": 512,
          "attention_k_eq_v": True,
          "sliding_window": 1024,
          "max_position_embeddings": 262144,
          "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
          "rope_parameters": {
              "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
              "sliding_attention": {"rope_theta": 10000.0},
          },
      },
  }
  ```

  Required tests:

  ```python
  def test_12b_real_config_detection(tmp_path: Path) -> None:
      (tmp_path / "config.json").write_text(json.dumps(GEMMA4_12B_CONFIG))
      assert _detect_gemma4_variant(tmp_path) is Gemma4Variant.DENSE_12B_UNIFIED

  def test_12b_real_config_overrides(tmp_path: Path) -> None:
      (tmp_path / "config.json").write_text(json.dumps(GEMMA4_12B_CONFIG))
      overrides, layer_types = _parse_gemma4_architecture(tmp_path)
      assert overrides["hidden_size"] == 3840
      assert overrides["head_dim"] == 256
      assert overrides["global_head_dim"] == 512
      assert overrides["num_global_key_value_heads"] == 1
      assert overrides["max_seq_len"] == 262144
      assert overrides["window_size"] == 1024
      assert overrides["partial_rotary_factor"] == 0.25
      assert overrides["image_token_id"] == 258880
      assert overrides["audio_token_id"] == 258881
      assert layer_types == [0, 0, 0, 0, 0, 1]
  ```

  Expected red: `_parse_gemma4_architecture()` returns defaults from top-level
  keys and maps `"full_attention"` as sliding.

- [ ] 1.2 Normalize nested Gemma 4 configs in Python.

  Target: modify `src/py/mogemma/model.py:153-227`.

  Add helpers above `_parse_gemma4_architecture()`:

  ```python
  def _gemma4_text_config(config: dict[str, object]) -> dict[str, object]:
      text_config = config.get("text_config")
      if isinstance(text_config, dict):
          return text_config
      return config

  def _layer_type_to_int(layer_type: object) -> int:
      return 1 if layer_type in {"full", "full_attention"} else 0
  ```

  Then change `_parse_gemma4_architecture()` to read nested `text_config` for
  text geometry while preserving top-level token ids:

  ```python
  text_config = _gemma4_text_config(config)
  window_size = text_config.get("sliding_window_size", text_config.get("sliding_window", 1024))
  rope_params = text_config.get("rope_parameters", {})
  full_rope = rope_params.get("full_attention", {}) if isinstance(rope_params, dict) else {}
  partial_rotary_factor = full_rope.get("partial_rotary_factor", text_config.get("partial_rotary_factor", 0.5))
  overrides["head_dim"] = int(text_config.get("head_dim", 0))
  if text_config.get("global_head_dim") is not None:
      overrides["global_head_dim"] = int(text_config["global_head_dim"])
  if text_config.get("num_global_key_value_heads") is not None:
      overrides["num_global_key_value_heads"] = int(text_config["num_global_key_value_heads"])
  if text_config.get("max_position_embeddings") is not None:
      overrides["max_seq_len"] = int(text_config["max_position_embeddings"])
  ```

  Also accept top-level `image_token_id` / `audio_token_id` in addition to the
  existing `image_token_index` / `audio_token_index`.

- [ ] 1.3 Checkpoint.

  Run:

  ```bash
  uv run pytest -q src/py/tests/test_gemma4_12b_config.py src/py/tests/test_variant_detection.py
  uv run ruff check src/py/mogemma/model.py src/py/tests/test_gemma4_12b_config.py
  ```

  Acceptance: 12B real-config tests pass and existing variant tests remain
  green.

### Phase 2: Add CPU Variable-Head Runtime Geometry

- [ ] 2.1 Add a failing Mojo/Python contract for mixed layer head dims.

  Target: extend `src/py/tests/test_mojo_core_gemma4_contract.py:52-84`.

  Add a two-layer tiny dense fixture:

  ```python
  def _tiny_12b_like_arrays() -> dict[str, np.ndarray]:
      arrays = {
          "model.embed_tokens.weight": np.ones((16, 8), dtype=np.float32),
          "model.norm.weight": np.ones((8,), dtype=np.float32),
          "lm_head.weight": np.ones((16, 8), dtype=np.float32),
      }
      # layer 0 sliding: 2 heads x dim 2, 1 kv head x dim 2
      # layer 1 full: 2 heads x dim 4, 1 global kv head x dim 4
      ...
      return arrays
  ```

  Required assertion:

  ```python
  llm = core.init_model_with_options(
      _meta(arrays),
      {
          "window_size": 4,
          "max_seq_len": 8,
          "layer_types": [0, 1],
          "head_dim": 2,
          "global_head_dim": 4,
          "num_global_key_value_heads": 1,
      },
      CPU_DESCRIPTOR,
  )
  assert llm["head_dim"] == 2
  assert llm["global_head_dim"] == 4
  assert list(llm["layer_head_dims"]) == [2, 4]
  assert list(llm["layer_kv_strides"]) == [2, 4]
  ```

  Expected red: `llm` has no `global_head_dim`, `layer_head_dims`, or
  `layer_kv_strides`.

- [ ] 2.2 Extend `KVCacheTrait` and CPU `KVCache` for per-layer geometry.

  Targets:

  - `src/mo/mogemma/model.mojo:25-48`
  - `src/mo/mogemma/model.mojo:407-561`

  Required shape:

  ```mojo
  trait KVCacheTrait:
      def get_layer_head_dim(self, layer: Int) -> Int: ...
      def get_layer_kv_stride(self, layer: Int) -> Int: ...
  ```

  Add fields to `KVCache`:

  ```mojo
  var layer_head_dims: List[Int]
  var layer_kv_strides: List[Int]
  var max_head_dim: Int
  ```

  Update `KVCache.__init__` signature to accept:

  ```mojo
  layer_head_dims_ptr: UnsafePointer[Int64, MutExternalOrigin]
  layer_kv_heads_ptr: UnsafePointer[Int64, MutExternalOrigin]
  ```

  Compute per-layer offsets with:

  ```mojo
  var layer_head_dim = Int(layer_head_dims_ptr.load(i))
  var layer_kv_heads = Int(layer_kv_heads_ptr.load(i))
  var kv_stride = layer_kv_heads * layer_head_dim
  self.layer_head_dims[i] = layer_head_dim
  self.layer_kv_strides[i] = kv_stride
  total_elements += cache_size * kv_stride
  ```

  Update `write_kv()` and `total_elements()` to use
  `self.layer_kv_strides[layer]`, not one global stride.

- [ ] 2.3 Build layer geometry arrays in Mojo init.

  Target: modify `src/mo/mogemma/core.mojo:666-789`.

  Required behavior:

  - Keep `head_dim` as the sliding/default head dim.
  - Add `global_head_dim` defaulting to `head_dim`.
  - Add `num_global_kv_heads` defaulting to `num_kv_heads`.
  - Build `layer_head_dims_np` and `layer_kv_heads_np` as `np.int64`.
  - Full layers use `global_head_dim` and `num_global_kv_heads`; sliding layers
    use `head_dim` and `num_kv_heads`.
  - Store both arrays on `llm` to keep backing memory alive:

    ```mojo
    py_dict["global_head_dim"] = global_head_dim
    py_dict["num_global_kv_heads"] = num_global_kv_heads
    py_dict["layer_head_dims"] = layer_head_dims_np
    py_dict["layer_kv_strides"] = layer_kv_strides_np
    ```

- [ ] 2.4 Split RoPE table geometry for sliding vs full attention.

  Targets:

  - `src/mo/mogemma/model.mojo:564-645`
  - `src/mo/mogemma/layers.mojo:220-264`

  Required shape:

  ```mojo
  struct RoPETables(Movable):
      var sliding_head_dim: Int
      var full_head_dim: Int
      var full_rotary_dim: Int
  ```

  Constructor signature:

  ```mojo
  def __init__(
      out self,
      sliding_head_dim: Int,
      full_head_dim: Int,
      partial_rotary_factor: Float32,
      window_size: Int,
      max_context_len: Int,
      theta_sliding: Float32 = 10000.0,
      theta_full: Float32 = 1000000.0,
  ):
  ```

  Sliding tables use `sliding_head_dim`; full tables use `full_head_dim` and
  `full_rotary_dim = even(full_head_dim * partial_rotary_factor)`.

- [ ] 2.5 Thread per-layer head dim through attention dispatch.

  Targets:

  - `src/mo/mogemma/layers.mojo:835-930`
  - `src/mo/mogemma/layers.mojo:79-187`
  - `src/mo/mogemma/layers.mojo:209-324`

  In `forward_gemma4_layer()`, compute:

  ```mojo
  var layer_head_dim = kv_cache.get_layer_head_dim(layer_idx)
  ```

  Pass `layer_head_dim` into `forward_sliding_attention()` and
  `forward_full_attention()`. Keep `hidden_size`, `num_heads`, and
  `num_kv_heads` as existing arguments for non-12B variants; use
  `kv_cache.get_layer_kv_stride(layer_idx)` inside cache write/read math.

  Acceptance detail: full-attention output projection input size must match
  `weights.o_proj.shape_1`, so the implementation should prefer tensor shapes
  over config-derived `num_heads * head_dim` when available.

- [ ] 2.6 Gate GPU for variable-head 12B.

  Targets:

  - `src/mo/mogemma/core.mojo:911-1018`
  - `src/py/tests/test_mojo_core_gemma4_contract.py`

  Required behavior: if `device_backend == "gpu"` and any layer head dim differs
  from `head_dim`, raise:

  ```text
  GPU runtime for Gemma 4 12B variable-head attention is not implemented
  ```

  Do not update `GPUKVCache` in this flow unless CPU variable-head contracts are
  already green and the change remains small.

- [ ] 2.7 Checkpoint.

  Run:

  ```bash
  make build
  uv run pytest -q src/py/tests/test_mojo_core_gemma4_contract.py
  uv run pytest -q src/mo/tests/test_mojo.py
  ```

  Acceptance: mixed sliding/full head-dim init passes on CPU; GPU requests are
  explicitly gated.

### Phase 3: Enable 12B Text-Only Initialization

- [ ] 3.1 Add tests for removing the blanket 12B runtime gate.

  Targets:

  - `src/py/tests/test_variant_detection.py:122-142`
  - `src/py/tests/test_gemma4_12b_config.py`

  Replace the old blanket-gate expectation with:

  ```python
  def test_12b_text_runtime_reaches_backend_after_geometry_support(tmp_path: Path) -> None:
      (tmp_path / "config.json").write_text(json.dumps(GEMMA4_12B_CONFIG))
      backend = _FakeBackend(return_value={"arch": "gemma4", "variant": "gemma4_dense_12b_unified"})
      _initialize_llm(
          _FakeLoader(model_path=tmp_path),
          backend,
          device_selection=resolve_device_selection("cpu"),
          model_type="generation",
          model_path=tmp_path,
      )
      assert backend.init_called is True
  ```

  Keep a separate test that 12B embeddings remain unsupported unless a concrete
  embedding contract is added in this flow.

- [ ] 3.2 Change Python runtime gating from model-wide to modality/device-specific.

  Targets:

  - `src/py/mogemma/model.py:142-150`
  - `src/py/mogemma/model.py:640-658`
  - `src/py/mogemma/model.py:702-753`

  Required behavior:

  - `_raise_for_unsupported_gemma4_runtime()` no longer rejects 12B text CPU
    initialization.
  - `SyncGemmaModel.__init__` stores `self._variant = _detect_gemma4_variant(self.model_path)`.
  - `generate_stream(..., images=...)` raises for 12B before `ImageHydrator()`:

    ```python
    if self._variant is Gemma4Variant.DENSE_12B_UNIFIED and images is not None:
        msg = "Gemma 4 12B unified image input is recognized but not implemented."
        raise RuntimeError(msg)
    ```

  - Existing audio rejection remains.

- [ ] 3.3 Update support matrix after CPU text contracts pass.

  Target: modify `src/py/mogemma/model_support.py:79-95`.

  Change only 12B text status:

  ```python
  text=RuntimeSupport.SUPPORTED,
  image=RuntimeSupport.UNSUPPORTED,
  audio=RuntimeSupport.UNSUPPORTED,
  unified_multimodal=RuntimeSupport.REQUIRES_FOLLOWUP,
  notes="Text-only CPU runtime is implemented; unified image/audio inputs remain explicitly gated.",
  ```

  Add assertions to `src/py/tests/test_model_support.py`.

- [ ] 3.4 Checkpoint.

  Run:

  ```bash
  uv run pytest -q src/py/tests/test_gemma4_12b_config.py src/py/tests/test_variant_detection.py src/py/tests/test_model_support.py
  uv run pytest -q src/py/tests/test_audio.py src/py/tests/test_vision_config.py
  ```

  Acceptance: 12B text support is reflected in metadata; image/audio remain
  explicit unsupported paths.

### Phase 4: Inventory and Loader Boundaries

- [ ] 4.1 Add a 12B local checkpoint inventory command artifact when weights are available.

  Target artifact: `.agents/specs/gemma4-12b-unified-runtime/12b-inventory.txt`.

  Use an opt-in local path from the user or environment; do not download 12B in
  default tests. Command shape:

  ```bash
  MOGEMMA_12B_LOCAL=/path/to/gemma-4-12B-it uv run python - <<'PY' > .agents/specs/gemma4-12b-unified-runtime/12b-inventory.txt
  import os
  from pathlib import Path

  from safetensors import safe_open

  root = Path(os.environ["MOGEMMA_12B_LOCAL"])
  print(root)
  for path in sorted(root.glob("*.safetensors")):
      with safe_open(path, framework="numpy") as handle:
          for name in handle.keys():
              tensor = handle.get_tensor(name)
              print(f"{path.name}:{name}:{tuple(tensor.shape)}:{tensor.dtype}")
  PY
  ```

  If no checkpoint is available, record `blocked: no local 12B checkpoint` in
  `learnings.md` and keep support scoped to synthetic + config contracts.

- [ ] 4.2 Add loader tests for 12B local safetensors only.

  Targets:

  - `src/py/tests/test_loader.py`
  - `src/py/tests/test_hub.py`
  - `src/py/mogemma/hub.py:22-35`

  Required assertions:

  - `KNOWN_GCS_MODELS` does not include `google/gemma-4-12B-it` unless a live
    GCS probe proves it.
  - A local directory with `config.json` and `model.safetensors` is accepted by
    `auto_loader()`.
  - A remote model id with no GCS prefix still raises `ModelNotFoundError` with
    the existing strict resolution path.

- [ ] 4.3 Guard Orbax conversion from claiming 12B support without inventory.

  Targets:

  - `src/py/mogemma/convert.py:34-53`
  - `src/py/tests/test_convert.py:760-841`

  Required behavior:

  ```python
  with pytest.raises(ValueError, match="Gemma 4 12B unified Orbax conversion requires a validated tensor inventory"):
      _variant_from_keys(["unified_encoder.some_new_tensor"])
  ```

  If a real 12B Orbax inventory is available, replace this gate with a real
  `"unified"` variant plan and tests in a separate flow.

- [ ] 4.4 Checkpoint.

  Run:

  ```bash
  uv run pytest -q src/py/tests/test_loader.py src/py/tests/test_hub.py src/py/tests/test_convert.py
  ```

  Acceptance: local-safetensors support boundary is explicit; remote/download
  support is not overclaimed.

### Phase 5: Verification, Docs, and Flow Closeout

- [ ] 5.1 Update docs for 12B text-only support.

  Targets:

  - `README.md`
  - `.agents/knowledge/gemma4-models.md`
  - `.agents/knowledge/python-runtime.md`
  - `.agents/flows.md`

  Required wording:

  - 12B is official.
  - 12B local text runtime is supported only after this flow's runtime tests
    pass.
  - 12B image/audio unified inputs remain follow-up.
  - 12B is not listed as GCS-downloadable unless live-probed.

- [ ] 5.2 Full verification.

  Run:

  ```bash
  make build
  make lint
  uv run pytest -q src/py/tests
  uv run pytest -q src/mo/tests/test_mojo.py
  ```

  Acceptance: all commands exit 0. If `make lint` emits Mojo deprecation
  warnings but exits 0, record that in `learnings.md` rather than blocking this
  flow.

- [ ] 5.3 Beads and archive readiness.

  Run:

  ```bash
  bd show mogemma-cpx
  bd list --parent mogemma-cpx --status open
  bd sync --flush-only
  ```

  Acceptance:

  - All child tasks are closed or explicitly blocked/deferred.
  - Flow spec, Beads status, and `.agents/flows.md` agree.

## Open Decisions

- Real 12B checkpoint inventory is still needed before claiming live parity.
  The synthetic contract can validate geometry, but not tensor-name compatibility.
- GPU support for variable-head 12B attention is intentionally gated in this
  flow unless CPU support lands cleanly and GPU changes remain mechanical.
- Unified 12B image/audio support is a follow-on Flow after text runtime and
  local weight inventory are stable.
