# Research: Gemma-Data GCS Bucket Model Formats

## Executive Summary

Google's public GCS bucket `gs://gemma-data` distributes Gemma model checkpoints primarily as **Orbax (OCDBT) format** checkpoints, rather than Hugging Face `.safetensors`, PyTorch `.bin`/`.pt`, or GGUF formats. Tokenizers are distributed as raw SentencePiece `.model` files.

The checkpoints discovered under `gs://gemma-data/checkpoints/` include multiple variants of `gemma2` and `gemma3` (e.g., `gemma3-4b-it`, `gemma3-27b-pt`, `gemma3n-e2b-it`). All are consistently saved as Orbax repositories characterized by `manifest.ocdbt` and `ocdbt.process_0` subdirectories.

## Codebase Analysis

The `mogemma` project already acknowledges this format and has an implementation in place to handle it:
- **`src/py/mogemma/convert.py`**: Contains the logic to convert Orbax/OCDBT checkpoints to HuggingFace-style `.safetensors` format.
- **`src/py/mogemma/orbax_loader.py`**: Used by the conversion script to parse the key-value store containing the Zarr-based array shards.

The backend Mojo engine expects contiguous, Hugging Face-named weights stored in `.safetensors` format, so this transformation acts as a bridge after initial download.

## Library Documentation

- **Orbax/OCDBT**: An open-source checkpointing library used heavily within the JAX/Flax ecosystem. It organizes tensors into a scalable key-value directory structure utilizing `tensorstore` and Zarr arrays underneath.
- **Safetensors**: The target format used by `mogemma`, chosen for its zero-copy memory-mapping capabilities.

## Prior Art

The repository's current `convert.py` and `orbax_loader.py` correctly identify the need to map OCDBT-formatted tensors into HF-compatible names and structures (e.g., reshaping or merging queries/keys/values depending on the specific model architecture layout).

## Risk Assessment

1. **Memory Overhead**: The `gemma2-27b` and `gemma3-27b` variants are very large. Converting Orbax checkpoints to `.safetensors` locally can result in OOM (Out Of Memory) exceptions if the conversion script attempts to load all tensors into RAM simultaneously.
2. **Format Instability**: Orbax representations can evolve, and the naming conventions for internal tensor arrays (`gating_einsum`, `q_einsum`, `per_layer_input_embedding`, etc.) vary wildly between standard Gemma 3 and Gemma 3 Nano variants. Any changes in Google's export pipeline could break the hardcoded mappings in `convert.py`.
3. **Download Friction**: Users downloading raw OCDBT formats must pull thousands of smaller chunk files, which can be slower than downloading a few large `.safetensors` files from Hugging Face Hub.

## Recommended Approach

1. **Continue Streamlined Conversion**: Retain the current `OrbaxLoader` strategy but ensure the conversion script processes tensors layer-by-layer (or shard-by-shard) to prevent memory spikes on consumer hardware, particularly when processing 27B parameter models.
2. **Architecture Dispatching**: Ensure that `convert.py` continues to dynamically detect the model family (e.g. standard Gemma 3 vs. Gemma 3 Nano) based on tensor metadata keys rather than relying strictly on the folder name.
3. **Fallback Options**: Because parsing Orbax can be brittle, offer Hugging Face Hub (`.safetensors` direct download) as an alternative or primary delivery mechanism where feasible, relegating the GCS bucket conversion to a specialized offline or developer path.
