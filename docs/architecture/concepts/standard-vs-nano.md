# Standard vs. Nano Architectures

Gemma 3 introduces a distinct structural variation for its smallest models: the **Nano** architecture. The mogemma inference engine implements both the Standard and Nano pathways to support the full model family.

## Standard Architecture

The standard Gemma 3 model uses a conventional transformer decoder structure.

**Data Flow:**
1. **Embedding**: The token ID is embedded into a high-dimensional vector.
2. **Transformer Layers**: The vector passes sequentially through $N$ transformer layers.
   - Each layer consists of a **Self-Attention** block (with RoPE and KV caching) and a **Feed-Forward Network (MLP)** block.
   - Residual connections and RMSNorm are applied at each step.
3. **Final Norm & Un-embedding**: The final vector is normalized and multiplied by the `lm_head` to produce logits over the vocabulary.

This linear, uniform structure means every token passes through identical layer operations with different learned weights.

## Nano Architecture (AltUp & Laurel)

The Gemma 3 Nano models (e.g., the 4B variant) heavily modify the layer structure to achieve high performance with a much smaller memory footprint. The Nano architecture introduces several novel mechanisms:

### 1. AltUp (Alternating Updates)
Instead of a single continuous residual stream of `hidden_size`, the Nano model splits its representational capacity into multiple "modalities" or streams (typically 4 streams, each of `hidden_size`). 

- **Prediction & Correction**: At each layer, an AltUp router predicts how the streams will evolve, routes one specific stream (the "active" stream) through the heavy mathematical operations (Attention and MLP), and then uses learned correction coefficients to update the other streams based on the result.
- **Why?** This allows the model to maintain a very wide effective hidden state without paying the computational cost of running full Attention and MLPs on the entire state.

### 2. Laurel
Before the active stream enters the standard Attention/MLP blocks, it passes through the Laurel module.
- Laurel consists of a down-projection and an up-projection. It acts as an efficient bottleneck/mixer before the main layer computations.

### 3. Per-Layer Mapping
Because the token embedding is injected into a complex multi-stream state, the Nano architecture includes a `per_layer_embed` mechanism. 
- The token is re-injected or modified via a per-layer mapping at the start of *every* layer, rather than just once at the beginning of the model. This helps the model retain strong input signal despite the aggressive AltUp routing.

### 4. KV Cache Sharing and Sparsity
To further save memory, Nano models employ **KV Sharing** across layers.
- While the standard model writes new Key and Value tensors to the cache at every layer, the Nano model stops writing new KV entries after a certain layer index (the `kv_share_start` boundary). 
- Subsequent layers reuse the cached KV tensors from earlier layers (often sliding or pinning to specific earlier layers). This drastically reduces the VRAM required to store the context of long sequences.

## Summary

When initializing a model, mogemma automatically detects the architecture by looking for Nano-specific weight keys (like `altup.router.weight`). 
- The **Standard** engine path loops over `forward_layer`.
- The **Nano** engine path loops over `forward_nano_layer`, which internally orchestrates stream prediction, Laurel, Attention, MLP, and AltUp correction.