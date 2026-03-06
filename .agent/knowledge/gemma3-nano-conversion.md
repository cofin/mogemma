# Knowledge: Gemma3 Nano Conversion Pipeline

> Flow: gemma3-nano-conversion | Archived 2026-02-25

## Key Changes

Built `_convert_gemma3_nano()` for Nano's 671-tensor Orbax layout. Added model family auto-detection. Extracted shared conversion helpers (`_convert_attention_layer`, `_convert_mlp_layer`, `_convert_norms`).

## Patterns

- **Model family detection:** Presence of `altup` or `per_layer_mapping` tensor keys distinguishes Nano from standard.
- **Shared conversion logic:** DRY extraction for attention, MLP, and norm conversions.
- **3D per-layer embeddings:** 262144x30x256 tensor handled as single contiguous block.

## Gotchas

- Nano has 671 tensors vs standard Gemma 3's 237-341 — fundamentally different architecture.
- Nano dimensions: hidden=2048, heads=8, kv_heads=2, head_dim=256, layers=30.
- Attention uses `query_norm` (not `_query_norm` like standard) — naming inconsistency.
- Massive per-layer embedding table requires special handling.
