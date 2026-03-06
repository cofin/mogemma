# Flow: pure-mojo-transformer

## Specification

### Goal
Assemble the native Mojo math primitives developed in Chapter 2 into a complete Gemma 3 Decoder Block, including Grouped Query Attention (GQA). Write the sequence processing loop to process full sequences (`input_ids`), and implement mean/last-token pooling to extract high-quality embeddings.

### Code Analysis Summary
- `src/mo/ops.mojo` contains `rms_norm`, `geglu`, `rope_rotate`, and `vec_mat_mul`.
- `src/mo/core.mojo` currently accepts memory pointers for the `.safetensors` model weights. We need to define structs to manage the transformer weights (Embedding, Decoder Layers, Final Norm).
- We must build the full attention mechanism (Q, K, V projections, RoPE application, Softmax, output projection) and the feed-forward network (Gate/Up projections, GeGLU, Down projection).
- We must assemble a full forward pass that takes an array of `input_ids` and propagates them through all layers to produce a final embedding.

### Requirements
1. **Model Weights Structs**: Define Mojo structs (e.g., `GemmaModelWeights`, `DecoderLayerWeights`) that hold the unboxed tensor pointers and metadata.
2. **Attention Mechanism**: Implement Grouped Query Attention (GQA) where multiple query heads share a single KV head. Apply RoPE to queries and keys. Implement softmax over attention scores.
3. **Feed-Forward Network (MLP)**: Combine `vec_mat_mul` and `geglu` into the standard LLaMA/Gemma MLP block (Up + Gate -> GeGLU -> Down).
4. **Decoder Block**: Connect Attention and MLP with `rms_norm` and residual connections.
5. **Embedding Extractor**: Iterate over `input_ids`, look up initial embeddings, process through all layers, apply final `rms_norm`, and apply pooling (e.g., last token or mean pooling) to produce a dense representation.
6. **Tests**: Verify the full block or attention mechanism computes correct values against a reference implementation.

### Non-goals
- Auto-regressive generation (KV Cache updates, sampling) — this is Chapter 4 (`pure-mojo-generation`). We only need to process the prompt sequence to completion for embeddings.

## Implementation Plan

### Phase 1: Struct and Data Setup
- [x] 1.1 Define structs in `src/mo/model.mojo` to represent the nested weight hierarchy of Gemma 3.
- [x] 1.2 Implement utility to fetch initial embeddings based on `input_ids`.

### Phase 2: Attention Mechanism
- [x] 2.1 Implement `softmax` for sequence lengths.
- [x] 2.2 Implement Grouped Query Attention (GQA), projecting Q, K, V and computing `QK^T / sqrt(d)`.
- [x] 2.3 Apply `rope_rotate` to Q and K prior to multiplication.

### Phase 3: MLP and Decoder Block
- [x] 3.1 Implement the MLP block using the previously defined `geglu` and `vec_mat_mul`.
- [x] 3.2 Combine Attention and MLP with residual connections and `rms_norm` into a `forward_layer` function.

### Phase 4: Full Sequence Processing & Pooling
- [x] 4.1 Write a `forward_sequence` loop that iterates over `input_ids` and applies the layers iteratively.
- [x] 4.2 Implement final pooling (last-token or mean pooling) to produce the embedding vector.

### Phase 5: Testing & Integration
- [x] 5.1 Expand the Python reference script to compute a mock Attention and Decoder block.
- [x] 5.2 Add `src/mo/tests/test_model.mojo` to verify the attention and block logic.