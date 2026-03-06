# Specification - Text Generation Core

## Goal
Implement the core text inference engine for Gemma 3, including stateful KV caching and a Pythonic generation API.

## Scope
- Mojo-side inference loop with KV cache management.
- Support for 4B, 12B, and 27B text model weights.
- Sampling implementation: Temperature, Top-K, Top-P (Nucleus sampling).
- Python `GemmaModel` wrapper with streaming support.

## Technical Requirements
- Efficient memory reuse in Mojo for KV cache buffers.
- Python `Iterator` based streaming API.
- Native Mojo implementation of Softmax and sampling kernels.

## Success Criteria
- [ ] Successful text generation from local Gemma 3 weights.
- [ ] Generation speed benchmarks (tokens/sec) recorded.
- [ ] Streaming interface works without stalling the main thread.
