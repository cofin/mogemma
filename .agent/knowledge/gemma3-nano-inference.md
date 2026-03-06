# Knowledge: Gemma3 Nano Inference

> Flow: gemma3-nano-inference | Archived 2026-02-23

## Architecture

Nano adds three components beyond standard transformer decoder:
1. **AltUp Routing:** 4-way modality pathways with prediction/correction coefficients.
2. **Laurel Pathway:** 64-dim bottleneck compression (2048->64->2048).
3. **Per-Layer Embedding:** 262K x 30 x 256 shared embedding table, gated per-token.

## Patterns

- **Separate NanoLayerWeights:** Embeds standard `LayerWeights` and adds Nano-specific fields (avoids bloating standard path).
- **Architecture dispatch:** Check for `model.layers.0.altup.router.weight` in metadata; if present, use Nano path.
- **Per-layer access pattern:** For layer L, token T: `T * 30 * 256 + L * 256` into contiguous memory.

## Gotchas

- Nano has fundamentally different forward pass order: attention -> per-layer mapping -> MLP -> Laurel -> AltUp.
- AltUp router has 4-way prediction with learned correction — most complex addition.
