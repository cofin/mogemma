# Learnings: Gemma3 Nano Inference

## [2026-02-23] - Nano Mojo backend extensions

- **Implemented:** NanoLayerWeights, AltUp forward pass, Laurel layer logic, per-layer mapping, architecture-aware dispatch.
- **Learnings:**
  - Separate NanoLayerWeights struct (embeds standard LayerWeights) avoids bloating standard path.
  - Architecture dispatch: check for `model.layers.0.altup.router.weight` in metadata.
  - Per-layer embedding access: for layer L, token T, slice at `T * 30 * 256 + L * 256`.
  - Nano forward pass order: attention -> per-layer mapping -> MLP -> Laurel -> AltUp.
  - AltUp 4-way prediction with learned correction is the most complex addition.
