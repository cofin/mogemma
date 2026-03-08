# Flow: mojo-vision-preprocessing_20260308

## Specification

### Code Analysis Summary
**Files Analyzed**:
- `src/mo/mogemma/core.mojo`: Exposes Python entrypoints (like `step_mojo`, `generate_embeddings_mojo`) using the Python C-API bindings. Will need new entrypoints to handle image arrays.
- `src/mo/mogemma/ops.mojo`: Contains math primitives (RMSNorm, matmul). Image patchification, bicubic resizing, and standard ImageNet normalizations are missing.
- `src/mo/mogemma/layers.mojo`: Contains text transformer logic. Missing the Vision Transformer block sequence.

### Requirements
1. **Mojo Native Preprocessing**: Implement resizing (e.g. bicubic) and pixel normalization logic in `ops.mojo` (or a dedicated `vision_ops.mojo`) to transform raw `[H, W, C]` bytes into `[N, hidden_size]` patch embeddings without relying on Pillow/NumPy.
2. **Vision Transformer Forward Pass**: Create the `forward_vision_layer` function to execute SigLIP/Vision Transformer blocks over the patchified image.
3. **Core Entrypoint**: Expose a new function `process_image_mojo` via `PyInit__core()` in `core.mojo` to accept a raw byte array or flat list, process the image, execute the vision layers, and yield the resulting projected continuous tokens.

## Implementation Plan

### Phase 1: Native Vision Primitives
- [ ] Task 1.1: Create basic array operations for image resizing and normalization (e.g., converting RGB bytes to normalized float32 tensors) in `src/mo/mogemma/ops.mojo` or `vision_ops.mojo`.
- [ ] Task 1.2: Implement the image patchification kernel (convolutional equivalent) to extract patches and add 2D positional embeddings.

### Phase 2: Vision Tower Execution
- [ ] Task 2.1: Add `forward_vision_layer` to `src/mo/mogemma/layers.mojo` handling SigLIP-specific Self-Attention and MLP.
- [ ] Task 2.2: Add an orchestration function `_forward_vision_tower_runtime` in `src/mo/mogemma/core.mojo` that loops through all vision layers.

### Phase 3: Core Module Integration
- [ ] Task 3.1: Define and export `process_image_mojo` in `src/mo/mogemma/core.mojo` that takes a Python bytes/NumPy array object, executes the pipeline, and returns the multimodal token embedding tensor back to Python.