# Gemma 4 Architecture & Implementation

Gemma 4 introduces several architectural improvements and multimodal capabilities, including hybrid attention, per-layer embeddings (PLE), and dedicated vision/audio encoders.

## Model Family Variants

| Variant | Architecture | Modalities | Context | Window |
| --- | --- | --- | --- | --- |
| **E2B** | Dense + PLE | Text, Image, Audio | 128K | 512 |
| **E4B** | Dense + PLE | Text, Image, Audio | 128K | 512 |
| **26B A4B** | MoE (128 exp) | Text, Image | 256K | 1024 |
| **31B** | Dense | Text, Image | 256K | 1024 |

## Attention Mechanism

Gemma 4 uses a hybrid attention mechanism combining sliding-window and global attention.

- **Hybrid Layout:** Most layers use `sliding_attention`, while `full_attention` layers are interleaved at regular intervals. The last layer is always `full_attention`.
- **RoPE behavior:** Sliding layers use standard RoPE (`theta=10000`). Full layers use proportional RoPE (`theta=1000000`) with a partial rotary factor.
- **KVCache Layout:** A single contiguous arena for all layers is preferred to avoid allocation sprawl. Offsets are calculated per layer.
- **Ring Buffer:** Windowed attention is implemented as a ring buffer. Causal masking is straightforward as all windowed entries are valid once the buffer has wrapped.

## Multimodal Integration

Vision and audio capabilities are integrated through a unified FFI abstraction.

- **Step with Embedding:** Pre-computed vectors from encoders (Vision, Audio) are injected directly into the decoder pass, bypassing the token embedding lookup.
- **Vision Encoder (SigLIP):** Uses bidirectional attention and standard GELU. Patches are averaged to keep sequence lengths manageable. Patch size is 16.
- **Audio Encoder:** Reuses the vision encoder's transformer structure. Mel spectrograms are computed in Python and passed via FFI. (E2B/E4B only).

## Specialized Layer Features

- **Per-Layer Embeddings (PLE):** Hidden states are augmented with bottleneck-projected embeddings at the start of each layer (E2B/E4B).
- **Shared-KV:** Some layers reuse the K/V projections from previous layers, saving memory and compute.
- **Mixture-of-Experts (MoE):** High-expert counts (e.g., 128 experts, top-8 gating) require efficient top-k gating and zero-copy weight access via Safetensors mmap.

## Data Delivery (HuggingFace)

- **Hub Implementation:** Uses `obstore` with `HTTPStore` for authenticated downloads from HuggingFace.
- **Shard Discovery:** Since HF lacks bucket-style listing, `model.safetensors.index.json` is used for discovery.
- **bf16 Loading:** `bf16` tensors are converted to `f32` by left-shifting bits by 16. References must be kept in Python to prevent GC before Mojo consumption.
