# Flow: quantization-int8-support_20260308

## Specification

### Code Analysis Summary
**Files Analyzed**:
- `src/py/mogemma/convert.py`: Currently loads float32 arrays from Orbax and saves them directly as float32 in safetensors. We need to optionally detect if an 8-bit mode is requested, compute quantization scales, cast tensors to `np.int8`, and store scales alongside them.
- `src/mo/mogemma/ops.mojo`: Provides `vec_mat_mul` and `mat_mat_mul` for `Float32`. We need new signatures or quantized routines (e.g. `vec_mat_mul_i8`) that take `Int8` weights and `Float32` scales.
- `src/mo/mogemma/model.mojo`: Defines `TensorInfo` and struct architectures. We need a way to track if a tensor is quantized (e.g., adding `is_quantized` flag or splitting `weight` into `weight_data` and `weight_scale`).

### Requirements
1. **Conversion Support**: Add a CLI flag or config to `convert_orbax_to_safetensors` that triggers symmetric INT8 quantization on linear layers (e.g., `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). Compute per-channel or per-tensor scales and save them.
2. **Mojo Quantized Types**: Update `TensorInfo` or create a new `QuantizedTensorInfo` in `model.mojo` to hold both the raw `Int8` data pointer and the `Float32` scale pointer.
3. **Mojo Runtime Ops**: Implement `vec_mat_mul_i8` and `mat_mat_mul_i8` in `ops.mojo`. These functions will perform the math using the 8-bit weights, dequantize the accumulations using the scales, and output `Float32` activations.
4. **Core Integration**: Update `init_model_mojo` and layer definitions to route properly when quantized weights are detected.

## Implementation Plan

### Phase 1: Python Conversion Pipeline
- [ ] Task 1.1: Add `quantize_int8` flag to `convert.py`. When enabled, perform symmetric int8 quantization on Linear layer weights and store the `weight` as `int8` and a corresponding `weight_scale` as `float32`.
- [ ] Task 1.2: Add unit tests for the conversion logic to ensure scales and quantized values correctly reconstruct the original weights with minimal loss.

### Phase 2: Mojo Structs & Initialization
- [ ] Task 2.1: Update `TensorInfo` in `model.mojo` to include an optional `scale_ptr`.
- [ ] Task 2.2: Update `_tensor_from_meta` in `core.mojo` to identify `int8` vs `float32` tensors from safetensors metadata and attach the scale pointer.

### Phase 3: Mojo Quantized Ops
- [ ] Task 3.1: Implement `vec_mat_mul_i8` in `ops.mojo` that performs dequantization on the fly during vector-matrix multiplication.
- [ ] Task 3.2: Update `forward_layer` to dispatch to the `_i8` version of matmul if the `TensorInfo` indicates it is a quantized tensor.