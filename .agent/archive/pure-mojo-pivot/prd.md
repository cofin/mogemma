# Master PRD: Pure Mojo Inference Pivot

## Context

`mogemma` was envisioned as a lightweight, fastembed-like library for Gemma 3 inference using the Mojo programming language. However, the current architecture merely wraps the heavy Modular `max` Python framework. This pulls in a massive 1GB+ Python ecosystem, relies on the Python GIL, and ironically fails because MAX does not yet natively support the chosen `gemma3n-e2b-it` architecture.

**North Star:** Pivot the architecture to a Pure Mojo Inference Engine. Strip out all heavy dependencies (`modular`, `max`). Use `safetensors` in Python to memory-map model weights, pass raw memory pointers to Mojo, and implement the transformer math (matmul, RMSNorm, RoPE, GeGLU) natively in Mojo using SIMD and `linalg`. This achieves a microscopic footprint (~40MB) and true C++ level inference speeds.

## Roadmap

### Chapter 1: `pure-mojo-bridge`
**Dependency Purge & Zero-Copy Bridge**
Remove `modular` and `max` from `pyproject.toml`. Implement a Python weight loader that memory-maps `safetensors` and passes raw memory pointers to Mojo. Create the Mojo FFI layer to accept these pointers and cast them into zero-copy Mojo `Tensor` views.

### Chapter 2: `pure-mojo-primitives`
**Native Mojo Math Primitives**
Implement Gemma 3's specific `GeGLU` activation function, `RMSNorm`, and Rotary Positional Embeddings (RoPE) natively in Mojo. Set up optimized `linalg.matmul` operations for linear projections (Q/K/V, O, MLP).

### Chapter 3: `pure-mojo-transformer`
**Decoder Block & Embedding Forward Pass**
Assemble the primitives into a full Gemma 3 Decoder Block (with Grouped Query Attention). Write the sequence processing loop to process `input_ids`. Implement pooling to extract high-quality embeddings.

### Chapter 4: `pure-mojo-generation`
**KV-Cache & LLM Generation**
Create a Mojo structure to manage KV-Cache tensors in raw memory. Adapt the forward pass for incremental single-token decoding. Implement Temperature, Top-K, and Nucleus sampling natively in Mojo, and expose a Python streaming generator.

## Global Constraints

1. **Microscopic Footprint:** The Python package must only depend on `tokenizers`, `numpy`, `safetensors`, `obstore`, and core Python libs. No `max`, `modular`, or heavy ML frameworks.
2. **Zero-Copy:** Weights and KV-cache must not be copied between Python and Mojo. Pass memory pointers.
3. **Mojo Native Math:** All transformer math must be executed in Mojo, bypassing the Python GIL entirely.
4. **Testing:** Each primitive and block must be verified against expected Gemma 3 outputs for correctness.
