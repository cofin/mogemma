# Master PRD: Model Architecture Support

## Context

The mogemma Mojo inference engine was built for a single architecture (standard Gemma 3) with hardcoded dimension assumptions (`head_dim=256`, `embedding_dim=768`). After removing the HuggingFace CLI dependency and switching to direct GCS downloads with Orbax-to-safetensors conversion, two architectural gaps emerged:

1. **Standard Gemma 3 fragility:** Hardcoded dimensions in `core.mojo` will break if any standard Gemma 3 variant uses different head_dim or hidden_size. The default `GenerationConfig.model_path` still points to `gemma3n-e2b-it` which the Mojo backend cannot load.

2. **Gemma 3 Nano unsupported:** Nano uses a fundamentally different architecture (AltUp routing, Laurel layers, per-layer embeddings, 671 tensors) that requires both a new conversion pipeline and new Mojo layer implementations.

**North Star:** Make mogemma a robust, multi-architecture Gemma inference engine that dynamically adapts to model variant geometry and cleanly supports both standard Gemma 3 and Gemma 3 Nano.

## Roadmap

### Chapter 1: `gemma3-standard-hardening`
**Dynamic Dimensions & Standard Model Reliability**

Remove all hardcoded dimension assumptions from `core.mojo`. Compute `head_dim` from tensor shapes by default, with Python override support. Fix `GenerationConfig` default model. Add model variant detection and clear error messages for unsupported architectures. Validate end-to-end with gemma3-270m-it and gemma-3-1b-it.

### Chapter 2: `gemma3-nano-conversion`
**Nano Orbax-to-Safetensors Conversion Pipeline**

Build `_convert_gemma3_nano()` in `convert.py` that maps Nano's 671-tensor Orbax layout (AltUp projections, Laurel layers, per-layer embedding mappings) to HuggingFace-compatible safetensors. Add model-family auto-detection so `_ensure_safetensors()` picks the right converter.

### Chapter 3: `gemma3-nano-inference`
**Nano Mojo Backend Extensions**

Extend Mojo `model.mojo` with Nano-specific weight structures (AltUp routing weights, Laurel layer weights, per-layer embedding mappings). Implement AltUp forward pass and Laurel layer logic in `layers.mojo`. Wire the extended model into `core.mojo` with architecture-aware dispatch.

## Global Constraints

1. **Derive with Override:** Dimension parameters (head_dim, embedding_dim) must be computed from tensor shapes by default, with optional Python overrides for new architectures.
2. **Zero-Copy Preserved:** All weight passing remains pointer-based between Python and Mojo. No copies.
3. **Backward Compatible:** Standard Gemma 3 must continue working after each chapter. No regressions.
4. **Explicit Failures:** Unsupported architectures must produce clear `ValueError`/`Error` messages, never silent wrong results.
5. **Microscopic Footprint:** No new heavy dependencies. TensorStore already added for Orbax; no more.
