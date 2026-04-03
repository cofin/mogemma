# Chapter 6: E2B / E4B Dense + PLE + Audio

**Flow ID:** `gemma4-e2b-e4b`
**Parent PRD:** `gemma4-rewrite`
**Beads Epic:** `mogemma-kuil.6`
**Depends on:** Chapter 3 (attention engine), Chapter 5 (vision pipeline)
**Status:** Planned

---

## Goal

Implement Per-Layer Embedding (PLE), shared-KV decoder layers, double-wide MLP (E2B only), and the audio tower for the E2B and E4B Gemma 4 variants. These models share PLE and shared-KV mechanisms but differ in layer count, hidden size, and MLP width.

## Current Codebase State (Post-Ch4)

**What exists and is reusable:**
- `layers.mojo`: `forward_sliding_attention(... k_eq_v: Bool ...)` and `forward_full_attention(... k_eq_v: Bool ...)` — already support K=V flag. `forward_mlp()` — standard GEGLU. `forward_gemma4_layer()` — dispatches sliding/full + MLP. `forward_gemma4_step()` — single-token forward.
- `model.mojo`: `LayerWeights` (13 tensors), `ModelWeights`, `KVCache` (hybrid sliding/full), `RoPETables` (dual theta).
- `ops.mojo`: `geglu`, `rope_rotate`, `vec_mat_mul/mat_mat_mul` (f32 + i8), `rms_norm`, `softmax`.
- `core.mojo`: `_init_model_impl_mojo` parses `architecture_overrides` (window_size, partial_rotary_factor, k_eq_v, layer_types). `step_mojo` hydrates model and calls `forward_gemma4_step`.
- `hydration.py`: `ImageHydrator` — images only.
- `model.py`: `_detect_gemma4_variant` detects E2B/E4B via `hidden_size_per_layer_input` + `use_double_wide_mlp`. `_parse_gemma4_architecture` reads layer_types, window_size, etc.

**What doesn't exist:**
- No PLE weight structs or forward logic
- No shared-KV mechanism (distinct from deleted Nano KV-sharing — different architecture)
- No double-wide MLP variant
- No audio processing of any kind
- No audio weight structs

## Architecture Reference

### Per-Layer Embedding (PLE)

Each decoder layer augments the hidden state with a per-layer token embedding:
```
hidden_state += per_layer_gate[L] * per_layer_proj[L](per_layer_embed[L][token_id])
```

| Parameter | E2B | E4B |
|---|---|---|
| `hidden_size_per_layer_input` | 256 | 256 |
| `vocab_size_per_layer_input` | 262144 | 262144 |
| Projection | 256 → 1536 | 256 → 2560 |

HF tensor names:
```
model.layers.{i}.per_layer_input.per_layer_embedding.weight  [262144, 256]
model.layers.{i}.per_layer_input.per_layer_projection.weight [hidden_size, 256]
model.layers.{i}.per_layer_input.per_layer_norm.weight       [hidden_size]
```

### Shared-KV Layers

Suffix decoder layers reuse KV from a source layer's cache — only Q and O projections are fresh.

| Variant | Total Layers | Fresh-KV | Shared-KV | Source mapping |
|---|---|---|---|---|
| E2B | 35 | 0–14 | 15–34 | config.json `kv_sharing_layer_map` |
| E4B | 42 | 0–23 | 24–41 | config.json `kv_sharing_layer_map` |

### Double-Wide MLP

E2B only (`use_double_wide_mlp=true`): intermediate_size is 2× standard. E4B uses standard width.

### Audio Tower

- Input: 16kHz mono audio → mel spectrogram
- Feature extraction: STFT via `numpy.fft`, mel filterbank (numpy matrix math), log-mel
- `audio_ms_per_token=40` → one token per 40ms
- `audio_seq_length=750` → max 30 seconds
- Audio encoder projects features → decoder hidden space
- Placeholder tokens in text replaced with audio embeddings (same pattern as vision)
- **No new Python dependencies:** numpy.fft for STFT, stdlib `wave` for WAV I/O

### Variant Differences

| Parameter | E2B | E4B |
|---|---|---|
| Layers | 35 | 42 |
| Hidden size | 1536 | 2560 |
| Attention heads | 8 | 8 |
| KV heads | 1 | 2 |
| Shared-KV layers | 20 | 18 |
| Double-wide MLP | Yes | No |
| Vision encoder layers | 16 | 16 |
| Sliding window | 512 | 512 |
| Context | 128K | 128K |

---

## Implementation Plan

### Phase 1: PLE Weight Structs

#### Task 6.1: PLELayerWeights in model.mojo

**Files:** `src/mo/mogemma/model.mojo`

**Details:**
- `PLELayerWeights` struct:
  - `per_layer_embedding: TensorInfo` — [262144, 256]
  - `per_layer_projection: TensorInfo` — [hidden_size, 256]
  - `per_layer_norm: TensorInfo` — [hidden_size]
- Add `ple_layers: List[PLELayerWeights]` field to `ModelWeights`
  - Empty list when PLE is not used (31B, 26B) — zero overhead
  - Populated for E2B/E4B
- Add `has_ple: Bool` flag to `ModelWeights`

**Tests:** Struct construction, empty vs populated PLE list.

---

### Phase 2: PLE Forward Pass

#### Task 6.2: forward_ple_input in layers.mojo

**Files:** `src/mo/mogemma/layers.mojo`

**Details:**
- `forward_ple_input(out_ptr, token_id: Int, layer_idx: Int, ple_weights: PLELayerWeights, hidden_size: Int, ple_dim: Int, scratch_ptr)`:
  1. Look up `per_layer_embedding[token_id]` → [ple_dim] vector
  2. Project: `per_layer_projection @ embedding` → [hidden_size]
  3. Apply `per_layer_norm` (RMSNorm)
  4. Add to `out_ptr` (the hidden state): `hidden += ple_output`
- Called at the **start** of each layer in the E2B/E4B forward step, before attention
- Gated: if `model.has_ple` is false, skip entirely

**Tests:** PLE output changes hidden state, gate=0 produces unchanged state, correct dim.

---

### Phase 3: Shared-KV Attention

#### Task 6.3: Shared-KV config parsing

**Files:** `src/py/mogemma/model.py`, `src/mo/mogemma/model.mojo`

**Details:**
- `_parse_gemma4_architecture` reads `kv_sharing_layer_map` from config.json
  - Array of ints: `kv_sharing_layer_map[target_layer] = source_layer` (−1 if no sharing)
  - Pass as architecture_override to Mojo
- `KVSharingConfig` in model.mojo: `source_layers: List[Int]` — per-layer mapping (−1 = compute own KV)

**Tests:** Parse E2B config (20 shared), E4B config (18 shared), no sharing for 31B.

---

#### Task 6.4: forward_shared_kv_attention in layers.mojo

**Files:** `src/mo/mogemma/layers.mojo`

**Details:**
- When `kv_sharing_map[layer_idx] >= 0`:
  1. Compute Q projection only (skip K, V projections)
  2. Read K, V from `source_layer`'s KV cache (already populated)
  3. Apply attention as normal (softmax(QK^T/√d) @ V)
  4. O projection
- Integrate into `forward_gemma4_layer` — check sharing map before dispatching attention
- Works with both sliding and full attention modes

**Tests:** Shared-KV uses source cache, K/V not computed, output matches manual computation.

---

### Phase 4: Double-Wide MLP

#### Task 6.5: forward_double_wide_mlp in layers.mojo

**Files:** `src/mo/mogemma/layers.mojo`

**Details:**
- Same as `forward_mlp` but with `intermediate_size * 2`
- Reuse `geglu` from ops.mojo — just different weight dimensions
- Dispatch: if `use_double_wide_mlp` flag is set, use doubled intermediate_size
- Simplest approach: parameterize `forward_mlp` to accept `intermediate_size` (it already does) — just pass the doubled value from config
- `_parse_gemma4_architecture` reads `use_double_wide_mlp` and computes `intermediate_size` accordingly

**Tests:** Double-wide intermediate dims for E2B, standard for E4B.

---

### Phase 5: E2B/E4B Forward Step

#### Task 6.6: forward_gemma4_ple_step in layers.mojo

**Files:** `src/mo/mogemma/layers.mojo`

**Details:**
- Extension of `forward_gemma4_step` that adds:
  1. PLE injection at each layer start
  2. Shared-KV dispatch in attention
  3. Configurable MLP width
- Signature: `forward_gemma4_ple_step(out_logits_ptr, token_id, pos, model: ModelWeights, ..., kv_sharing: List[Int], scratch_ptr)`
- Flow: embed → scale → for each layer: PLE → (shared-KV or fresh) attention → MLP → final norm → LM head
- Consider: merge into `forward_gemma4_step` with conditional branches vs separate function
  - Recommendation: separate function — keeps 31B fast path clean, avoids branch noise

**Tests:** Full step with PLE + shared-KV produces correct output shape, E2B vs E4B differences.

---

### Phase 6: Core.mojo Wiring

#### Task 6.7: _init_model_impl_mojo extensions for E2B/E4B

**Files:** `src/mo/mogemma/core.mojo`

**Details:**
- When `architecture_overrides` includes PLE params:
  - Build `PLELayerWeights` list from metadata (tensor names: `model.layers.{i}.per_layer_input.*`)
  - Set `model.has_ple = True`
  - Parse `kv_sharing_layer_map` from overrides
- `step_mojo` dispatches to `forward_gemma4_ple_step` when `model.has_ple`
- New overrides: `hidden_size_per_layer_input`, `vocab_size_per_layer_input`, `use_double_wide_mlp`, `kv_sharing_layer_map`

**Tests:** Init with E2B overrides, init with E4B overrides, PLE weights populated.

---

### Phase 7: Audio Feature Extraction (Python)

#### Task 6.8: Mel spectrogram extraction

**Files:** `src/py/mogemma/audio.py` (new file)

**Details:**
- `load_audio(path_or_bytes) → ndarray[float32]`:
  - WAV: stdlib `wave` module, read PCM, convert to float32 [-1, 1], resample to 16kHz if needed
  - Mono: average channels if stereo
  - Truncate to 30s (480000 samples)
- `mel_spectrogram(audio: ndarray, sr=16000, n_fft=400, hop_length=160, n_mels=80) → ndarray`:
  1. STFT: `numpy.fft.rfft` with Hann window, hop_length frames
  2. Power spectrum: `|STFT|^2`
  3. Mel filterbank: construct triangular filter matrix (mel scale → Hz → bin mapping)
  4. Apply: `mel_filters @ power_spectrum`
  5. Log: `log(mel + 1e-6)`
  6. Output shape: `[n_mels, num_frames]`
- `extract_audio_features(path_or_bytes, max_seconds=30) → ndarray`:
  - Loads, computes mel spectrogram, pads/truncates to `audio_seq_length` frames

**Tests:** Known WAV → mel features (compare against reference), truncation, padding, mono conversion.

---

#### Task 6.9: AudioHydrator in hydration.py

**Files:** `src/py/mogemma/hydration.py`

**Details:**
- `AudioHydrator.hydrate(inputs) → list[AudioInput]`:
  - Accept: WAV file path, raw bytes, numpy float32 array
  - Return `AudioInput(features: ndarray, num_tokens: int)` dataclass
  - `num_tokens = min(num_frames, 750)` (audio_seq_length cap)
- Lazy import: `wave` from stdlib
- Error for non-WAV without `mogemma[audio]` extra

**Tests:** WAV path, bytes, numpy input, format error, length limits.

---

### Phase 8: Audio Encoder (Mojo)

#### Task 6.10: AudioTowerWeights and forward_audio_encoder

**Files:** `src/mo/mogemma/model.mojo`, `src/mo/mogemma/layers.mojo`

**Details:**
- `AudioTowerWeights` struct in model.mojo:
  - `conv_layers: List[TensorInfo]` — 1D convolution stack (feature extraction)
  - `encoder_layers: List[AudioEncoderLayerWeights]` — transformer layers
  - `projection: TensorInfo` — [decoder_hidden, audio_hidden]
  - `position_embedding: TensorInfo`
- `AudioEncoderLayerWeights`: q/k/v/o_proj, fc1, fc2, norms (same structure as VisionLayerWeights)
- `forward_audio_encoder(out_ptr, features_ptr, weights: AudioTowerWeights, num_frames, audio_config...)`:
  1. Conv feature extraction
  2. Add position embeddings
  3. Transformer layers (bidirectional, same as vision)
  4. Projection to decoder hidden

**Tests:** Encoder with synthetic weights, output shape, projection dim.

---

#### Task 6.11: Audio placeholder merging

**Files:** `src/py/mogemma/model.py`, `src/mo/mogemma/core.mojo`

**Details:**
- New FFI: `process_audio_mojo(llm, features_obj)` — runs audio encoder, stores embeddings
- In `generate_stream()`: detect `<audio>` placeholder tokens, replace with audio embeddings during prefill (same `step_with_embedding` from Ch5)
- `_parse_gemma4_architecture` reads `audio_token_index` from config.json
- `GenerationConfig` gains `audio: Sequence[str | Path | bytes | ndarray] | None` param in generate methods

**Tests:** Placeholder detection, audio token merging, position IDs.

---

### Phase 9: Integration & Tests

#### Task 6.12: End-to-end E2B/E4B integration tests

**Files:** `src/py/tests/test_e2b_e4b_config.py`, `src/py/tests/test_audio.py`, `src/mo/tests/test_ple.mojo`

**Details:**
- E2B config wiring: PLE params, shared-KV map, double-wide MLP flag, vision (16 layers), audio
- E4B config wiring: PLE params, shared-KV map, standard MLP, vision (16 layers), audio
- PLE correctness with synthetic weights
- Shared-KV correctness
- Audio feature extraction (known WAV reference)
- Variant differences: E2B double-wide vs E4B standard

**Verification:**
```bash
CI=true uv run pytest -x -v
grep -r "nano\|altup\|laurel" src/mo/mogemma/ --include="*.mojo"
grep -r "scipy\|librosa\|torchaudio" src/py/mogemma/
```

---

## Files Modified

| File | Changes |
|---|---|
| `src/mo/mogemma/model.mojo` | Add PLELayerWeights, AudioTowerWeights, AudioEncoderLayerWeights, extend ModelWeights |
| `src/mo/mogemma/layers.mojo` | Add forward_ple_input, forward_shared_kv_attention, forward_gemma4_ple_step, forward_audio_encoder |
| `src/mo/mogemma/core.mojo` | PLE weight building, audio FFI, E2B/E4B dispatch |
| `src/py/mogemma/model.py` | PLE/shared-KV/audio config parsing, audio placeholder merging |
| `src/py/mogemma/hydration.py` | Add AudioHydrator |
| `src/py/mogemma/config.py` | Audio params in GenerationConfig |

## New Files

| File | Purpose |
|---|---|
| `src/py/mogemma/audio.py` | Mel spectrogram, WAV loading, audio feature extraction |
| `src/py/tests/test_audio.py` | Audio feature extraction tests |
| `src/py/tests/test_e2b_e4b_config.py` | E2B/E4B config and variant tests |
| `src/mo/tests/test_ple.mojo` | PLE forward, shared-KV, double-wide MLP |

## Task Summary

| Task | Description | Status |
|---|---|---|
| 6.1 | PLELayerWeights struct | [x] 379e65d |
| 6.2 | forward_ple_input | [x] 379e65d |
| 6.3 | Shared-KV config parsing | [x] 379e65d |
| 6.4 | forward_shared_kv_attention | [x] 379e65d (integrated into forward_gemma4_ple_step) |
| 6.5 | Double-wide MLP | [x] 379e65d |
| 6.6 | forward_gemma4_ple_step | [x] 379e65d |
| 6.7 | Core.mojo E2B/E4B wiring | [x] 74fc44f |
| 6.8 | Mel spectrogram extraction | [x] bdf0d64 |
| 6.9 | AudioHydrator | [x] c969f0d |
| 6.10 | AudioTowerWeights + forward_audio_encoder | [x] 66af24c |
| 6.11 | Audio placeholder merging | [x] d7b94c2 + 74fc44f |
| 6.12 | End-to-end integration tests | [x] 379e65d |
