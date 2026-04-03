"""GPU memory infrastructure for mogemma.

Provides GPUContext (DeviceContext lifecycle), buffer allocation helpers,
and host↔device transfer primitives. All GPU code is gated behind
`comptime if has_accelerator()` so this module compiles on CPU-only systems.
"""

from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.memory import UnsafePointer
from std.collections import List

from mogemma.model import (
    TensorInfo,
    LayerWeights,
    MoEExpertWeights,
    VisionLayerWeights,
    PtrPair,
    IntPair,
    LAYER_TYPE_SLIDING,
    LAYER_TYPE_FULL,
)


struct GPUContext(Movable):
    """Manages a GPU device context and provides buffer allocation/transfer helpers.

    Wraps `DeviceContext` from `std.gpu.host` with convenience methods for
    the mogemma weight-streaming protocol.
    """

    var ctx: DeviceContext
    var _alive: Bool

    def __init__(out self) raises:
        """Create a GPUContext, acquiring the default GPU device.

        Raises:
            Error if no GPU accelerator is available at runtime.
        """
        self.ctx = DeviceContext()
        self._alive = True

    def __moveinit__(out self, owned other: Self):
        self.ctx = other.ctx^
        self._alive = other._alive
        other._alive = False

    def allocate_buffer[dtype: DType](mut self, size: Int) raises -> DeviceBuffer[dtype]:
        """Allocate a DeviceBuffer of `size` elements on the GPU.

        Args:
            size: Number of elements to allocate.

        Returns:
            A DeviceBuffer residing in GPU memory.
        """
        return self.ctx.enqueue_create_buffer[dtype](size)

    def allocate_host_buffer[dtype: DType](mut self, size: Int) raises -> HostBuffer[dtype]:
        """Allocate a pinned HostBuffer of `size` elements for fast transfers.

        Args:
            size: Number of elements to allocate.

        Returns:
            A pinned HostBuffer for efficient host↔device copies.
        """
        return self.ctx.enqueue_create_host_buffer[dtype](size)

    def upload(mut self, mut dst: DeviceBuffer[DType.float32], src: HostBuffer[DType.float32]) raises:
        """Enqueue an async copy from pinned host memory to device memory.

        Args:
            dst: Target DeviceBuffer.
            src: Source HostBuffer (pinned).
        """
        self.ctx.enqueue_copy(dst, src)

    def download(mut self, mut dst: HostBuffer[DType.float32], src: DeviceBuffer[DType.float32]) raises:
        """Enqueue an async copy from device memory to pinned host memory.

        Args:
            dst: Target HostBuffer (pinned).
            src: Source DeviceBuffer.
        """
        self.ctx.enqueue_copy(dst, src)

    def sync(mut self) raises:
        """Block until all enqueued GPU operations complete."""
        self.ctx.synchronize()

    def cleanup(mut self) raises:
        """Release GPU resources. Safe to call multiple times."""
        if self._alive:
            self.ctx.synchronize()
            self._alive = False


struct WeightStage(Movable):
    """Reusable host+device staging buffers for layer-by-layer weight streaming.

    Holds a pinned HostBuffer and a DeviceBuffer, both sized to fit the largest
    layer's weights. Each layer step: memcpy mmap→host, enqueue host→device,
    compute, then overwrite staging for the next layer.
    """

    var device_buf: DeviceBuffer[DType.float32]
    var host_buf: HostBuffer[DType.float32]
    var capacity: Int  # max elements that fit
    var _offset: Int  # current write offset during packing

    def __init__(out self, mut ctx: GPUContext, capacity: Int) raises:
        """Allocate staging buffers with the given capacity.

        Args:
            ctx: GPU context for allocation.
            capacity: Max number of Float32 elements to stage.
        """
        self.device_buf = ctx.allocate_buffer[DType.float32](capacity)
        self.host_buf = ctx.allocate_host_buffer[DType.float32](capacity)
        self.capacity = capacity
        self._offset = 0

    def __moveinit__(out self, owned other: Self):
        self.device_buf = other.device_buf^
        self.host_buf = other.host_buf^
        self.capacity = other.capacity
        self._offset = other._offset

    def reset(mut self):
        """Reset the packing offset to reuse the staging buffer for a new layer."""
        self._offset = 0

    def _pack_tensor(mut self, tensor: TensorInfo) -> Int:
        """Copy a tensor's data from its mmap pointer into the host staging buffer.

        Returns the offset (in elements) where this tensor was packed.
        """
        var num_elements = tensor.shape_0 * tensor.shape_1
        if num_elements == 0:
            return self._offset
        var start = self._offset
        var dst = self.host_buf.unsafe_ptr() + start
        var src = tensor.ptr
        for i in range(num_elements):
            dst.store(i, src.load(i))
        self._offset += num_elements
        return start

    def upload_tensor(
        mut self, mut ctx: GPUContext, tensor: TensorInfo
    ) raises -> UnsafePointer[Float32, MutAnyOrigin]:
        """Upload a single tensor to the device staging buffer.

        Copies tensor data from its source pointer to the pinned host buffer,
        enqueues host→device copy, and returns the device pointer.

        Args:
            ctx: GPU context for the transfer.
            tensor: Source tensor with mmap pointer and shape.

        Returns:
            Device pointer into the staging buffer where the tensor resides.
        """
        self.reset()
        var start = self._pack_tensor(tensor)
        ctx.upload(self.device_buf, self.host_buf)
        return self.device_buf.unsafe_ptr() + start

    def upload_tensors(
        mut self, mut ctx: GPUContext, tensors: List[TensorInfo]
    ) raises -> List[UnsafePointer[Float32, MutAnyOrigin]]:
        """Upload multiple tensors packed contiguously into the staging buffer.

        Returns a list of device pointers, one per tensor.

        Args:
            ctx: GPU context for the transfer.
            tensors: Source tensors to pack and upload.

        Returns:
            List of device pointers corresponding to each tensor in the staging buffer.
        """
        self.reset()
        var offsets = List[Int]()
        for i in range(len(tensors)):
            offsets.append(self._pack_tensor(tensors[i]))
        ctx.upload(self.device_buf, self.host_buf)
        var ptrs = List[UnsafePointer[Float32, MutAnyOrigin]]()
        var base = self.device_buf.unsafe_ptr()
        for i in range(len(offsets)):
            ptrs.append(base + offsets[i])
        return ptrs


def _upload_persistent(
    mut ctx: GPUContext, tensor: TensorInfo
) raises -> DeviceBuffer[DType.float32]:
    """Upload a tensor to a dedicated persistent DeviceBuffer.

    Unlike WeightStage (reusable per layer), this allocates a permanent buffer
    that lives for the entire session.

    Args:
        ctx: GPU context for allocation and transfer.
        tensor: Source tensor to upload.

    Returns:
        A DeviceBuffer containing the tensor data.
    """
    var num_elements = tensor.shape_0 * tensor.shape_1
    var dev_buf = ctx.allocate_buffer[DType.float32](num_elements)
    var host_buf = ctx.allocate_host_buffer[DType.float32](num_elements)
    # Copy from mmap source to pinned host buffer
    var dst = host_buf.unsafe_ptr()
    var src = tensor.ptr
    for i in range(num_elements):
        dst.store(i, src.load(i))
    # Upload to device
    ctx.upload(dev_buf, host_buf)
    _ = host_buf  # host staging can be freed after sync
    return dev_buf^


struct PersistentBuffers(Movable):
    """Persistent GPU buffers for weights that are used every step.

    Holds embed_tokens, lm_head, and final norm on the GPU for the
    entire session. These are uploaded once at init and never overwritten.
    """

    var embed_buf: DeviceBuffer[DType.float32]
    var lm_head_buf: DeviceBuffer[DType.float32]
    var norm_buf: DeviceBuffer[DType.float32]
    var embed_ptr: UnsafePointer[Float32, MutAnyOrigin]
    var lm_head_ptr: UnsafePointer[Float32, MutAnyOrigin]
    var norm_ptr: UnsafePointer[Float32, MutAnyOrigin]
    var embed_elements: Int
    var lm_head_elements: Int
    var norm_elements: Int

    def __init__(
        out self,
        mut ctx: GPUContext,
        embed_tokens: TensorInfo,
        lm_head: TensorInfo,
        norm: TensorInfo,
    ) raises:
        """Upload embed_tokens, lm_head, and norm to persistent GPU buffers.

        Args:
            ctx: GPU context for allocation.
            embed_tokens: Embedding matrix (vocab_size × hidden_size).
            lm_head: LM head matrix (hidden_size × vocab_size).
            norm: Final RMS norm weights (hidden_size).
        """
        self.embed_elements = embed_tokens.shape_0 * embed_tokens.shape_1
        self.lm_head_elements = lm_head.shape_0 * lm_head.shape_1
        self.norm_elements = norm.shape_0 * norm.shape_1
        self.embed_buf = _upload_persistent(ctx, embed_tokens)
        self.lm_head_buf = _upload_persistent(ctx, lm_head)
        self.norm_buf = _upload_persistent(ctx, norm)
        ctx.sync()
        self.embed_ptr = self.embed_buf.unsafe_ptr()
        self.lm_head_ptr = self.lm_head_buf.unsafe_ptr()
        self.norm_ptr = self.norm_buf.unsafe_ptr()

    def __moveinit__(out self, owned other: Self):
        self.embed_buf = other.embed_buf^
        self.lm_head_buf = other.lm_head_buf^
        self.norm_buf = other.norm_buf^
        self.embed_ptr = other.embed_ptr
        self.lm_head_ptr = other.lm_head_ptr
        self.norm_ptr = other.norm_ptr
        self.embed_elements = other.embed_elements
        self.lm_head_elements = other.lm_head_elements
        self.norm_elements = other.norm_elements


struct GPUKVCache(Movable):
    """GPU-resident KV cache for Gemma 4 hybrid sliding-window + full attention.

    Same layout and offset math as CPU `KVCache`, but K/V storage lives in
    DeviceBuffers on the GPU. Metadata (layer types, offsets) stays on the host
    for orchestration; the actual cache data is accessed by GPU kernels via
    device pointers.
    """

    var num_layers: Int
    var num_kv_heads: Int
    var head_dim: Int
    var window_size: Int
    var max_context_len: Int

    # Per-layer metadata (host-side)
    var layer_types: List[UInt8]
    var layer_cache_sizes: List[Int]
    var layer_offsets: List[Int]

    # Device-resident storage
    var k_cache: DeviceBuffer[DType.float32]
    var v_cache: DeviceBuffer[DType.float32]
    var k_ptr: UnsafePointer[Float32, MutAnyOrigin]
    var v_ptr: UnsafePointer[Float32, MutAnyOrigin]

    def __init__(
        out self,
        mut ctx: GPUContext,
        num_layers: Int,
        num_kv_heads: Int,
        head_dim: Int,
        window_size: Int,
        max_context_len: Int,
        layer_types_ptr: UnsafePointer[UInt8, MutExternalOrigin],
    ) raises:
        """Allocate GPU KV cache with the same layout as CPU KVCache.

        Args:
            ctx: GPU context for buffer allocation.
            num_layers: Number of transformer layers.
            num_kv_heads: Number of KV attention heads.
            head_dim: Dimension per head.
            window_size: Sliding window size.
            max_context_len: Maximum context length for full attention layers.
            layer_types_ptr: Pointer to layer type array (LAYER_TYPE_SLIDING/FULL).
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.max_context_len = max_context_len

        # Build per-layer metadata (same logic as CPU KVCache)
        self.layer_types = List[UInt8](length=num_layers, fill=0)
        self.layer_cache_sizes = List[Int](length=num_layers, fill=0)
        self.layer_offsets = List[Int](length=num_layers, fill=0)

        var kv_stride = num_kv_heads * head_dim
        var total_elements: Int = 0
        for i in range(num_layers):
            var lt = layer_types_ptr.load(i)
            self.layer_types[i] = lt
            var cache_size: Int
            if lt == LAYER_TYPE_FULL:
                cache_size = max_context_len
            else:
                cache_size = window_size
            self.layer_cache_sizes[i] = cache_size
            self.layer_offsets[i] = total_elements
            total_elements += cache_size * kv_stride

        # Allocate device buffers (at least 1 element to avoid zero-size alloc)
        var alloc_size = total_elements if total_elements > 0 else 1
        self.k_cache = ctx.allocate_buffer[DType.float32](alloc_size)
        self.v_cache = ctx.allocate_buffer[DType.float32](alloc_size)
        self.k_ptr = self.k_cache.unsafe_ptr()
        self.v_ptr = self.v_cache.unsafe_ptr()

    def __moveinit__(out self, owned other: Self):
        self.num_layers = other.num_layers
        self.num_kv_heads = other.num_kv_heads
        self.head_dim = other.head_dim
        self.window_size = other.window_size
        self.max_context_len = other.max_context_len
        self.layer_types = other.layer_types^
        self.layer_cache_sizes = other.layer_cache_sizes^
        self.layer_offsets = other.layer_offsets^
        self.k_cache = other.k_cache^
        self.v_cache = other.v_cache^
        self.k_ptr = other.k_ptr
        self.v_ptr = other.v_ptr

    @always_inline
    def get_kv_ptrs(self, layer: Int) -> PtrPair:
        """Returns (k_ptr, v_ptr) pointing to the start of this layer's cache region on device."""
        var offset = self.layer_offsets[layer]
        var k = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(self.k_ptr + offset))
        var v = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(self.v_ptr + offset))
        return PtrPair(k, v)

    @always_inline
    def get_attention_range(self, layer: Int, pos: Int) -> IntPair:
        """Returns (valid_len, cache_size) for computing attention at the given position.

        Same logic as CPU KVCache: sliding layers cap at window_size, full layers grow linearly.
        """
        var cache_size = self.layer_cache_sizes[layer]
        var valid_len: Int
        if self.layer_types[layer] == LAYER_TYPE_FULL:
            valid_len = pos + 1
        else:
            if pos + 1 < cache_size:
                valid_len = pos + 1
            else:
                valid_len = cache_size
        return IntPair(valid_len, cache_size)

    def total_elements(self) -> Int:
        """Returns total Float32 elements allocated across all layers (per K or V)."""
        if self.num_layers == 0:
            return 0
        var last = self.num_layers - 1
        var kv_stride = self.num_kv_heads * self.head_dim
        return self.layer_offsets[last] + self.layer_cache_sizes[last] * kv_stride


struct GPUScratch(Movable):
    """GPU-resident scratch buffer for intermediate computation results.

    Sized using the same formula as the CPU scratch allocation in core.mojo.
    """

    var buf: DeviceBuffer[DType.float32]
    var ptr: UnsafePointer[Float32, MutAnyOrigin]
    var size: Int

    def __init__(
        out self,
        mut ctx: GPUContext,
        hidden_size: Int,
        max_seq_len: Int,
        num_heads: Int,
    ) raises:
        """Allocate GPU scratch buffer using the standard sizing formula.

        Args:
            ctx: GPU context for allocation.
            hidden_size: Model hidden dimension.
            max_seq_len: Maximum sequence length.
            num_heads: Number of attention heads.
        """
        # Match CPU scratch formula: max(step_scratch, embedding_scratch)
        var step_len = hidden_size * 160 + max_seq_len * num_heads * 2
        var emb_len = hidden_size * 180 + max_seq_len * num_heads * 2
        self.size = step_len if step_len > emb_len else emb_len
        self.buf = ctx.allocate_buffer[DType.float32](self.size)
        self.ptr = self.buf.unsafe_ptr()

    def __moveinit__(out self, owned other: Self):
        self.buf = other.buf^
        self.ptr = other.ptr
        self.size = other.size


# ── Layer Weight Upload Functions ─────────────────────────────────────────────


def _tensor_from_device_ptr(
    ptr: UnsafePointer[Float32, MutAnyOrigin], shape_0: Int, shape_1: Int
) -> TensorInfo:
    """Create a TensorInfo pointing to device memory."""
    return TensorInfo(Int(ptr), shape_0, shape_1)


def upload_layer_weights(
    mut stage: WeightStage, mut ctx: GPUContext, layer: LayerWeights
) raises -> LayerWeights:
    """Upload all tensors in a dense LayerWeights to the device staging buffer.

    Packs all 13 layer tensors contiguously into the staging buffer, performs
    one host→device copy, and returns a new LayerWeights with device pointers.

    Args:
        stage: Reusable staging buffer (reset and packed per call).
        ctx: GPU context for the transfer.
        layer: Source LayerWeights with CPU/mmap pointers.

    Returns:
        A new LayerWeights where all tensor pointers reference device memory.
    """
    stage.reset()

    # Pack all tensors into host staging buffer
    var q_off = stage._pack_tensor(layer.q_proj)
    var k_off = stage._pack_tensor(layer.k_proj)
    var v_off = stage._pack_tensor(layer.v_proj)
    var o_off = stage._pack_tensor(layer.o_proj)
    var gate_off = stage._pack_tensor(layer.gate_proj)
    var up_off = stage._pack_tensor(layer.up_proj)
    var down_off = stage._pack_tensor(layer.down_proj)
    var in_norm_off = stage._pack_tensor(layer.input_layernorm)
    var post_attn_off = stage._pack_tensor(layer.post_attention_layernorm)
    var q_norm_off = stage._pack_tensor(layer.q_norm)
    var k_norm_off = stage._pack_tensor(layer.k_norm)
    var pre_ff_off = stage._pack_tensor(layer.pre_feedforward_layernorm)
    var post_ff_off = stage._pack_tensor(layer.post_feedforward_layernorm)

    # Single host→device copy
    ctx.upload(stage.device_buf, stage.host_buf)

    # Build new LayerWeights with device pointers
    var base = stage.device_buf.unsafe_ptr()
    var result = LayerWeights()
    result.q_proj = _tensor_from_device_ptr(base + q_off, layer.q_proj.shape_0, layer.q_proj.shape_1)
    result.k_proj = _tensor_from_device_ptr(base + k_off, layer.k_proj.shape_0, layer.k_proj.shape_1)
    result.v_proj = _tensor_from_device_ptr(base + v_off, layer.v_proj.shape_0, layer.v_proj.shape_1)
    result.o_proj = _tensor_from_device_ptr(base + o_off, layer.o_proj.shape_0, layer.o_proj.shape_1)
    result.gate_proj = _tensor_from_device_ptr(base + gate_off, layer.gate_proj.shape_0, layer.gate_proj.shape_1)
    result.up_proj = _tensor_from_device_ptr(base + up_off, layer.up_proj.shape_0, layer.up_proj.shape_1)
    result.down_proj = _tensor_from_device_ptr(base + down_off, layer.down_proj.shape_0, layer.down_proj.shape_1)
    result.input_layernorm = _tensor_from_device_ptr(base + in_norm_off, layer.input_layernorm.shape_0, layer.input_layernorm.shape_1)
    result.post_attention_layernorm = _tensor_from_device_ptr(base + post_attn_off, layer.post_attention_layernorm.shape_0, layer.post_attention_layernorm.shape_1)
    result.q_norm = _tensor_from_device_ptr(base + q_norm_off, layer.q_norm.shape_0, layer.q_norm.shape_1)
    result.k_norm = _tensor_from_device_ptr(base + k_norm_off, layer.k_norm.shape_0, layer.k_norm.shape_1)
    result.pre_feedforward_layernorm = _tensor_from_device_ptr(base + pre_ff_off, layer.pre_feedforward_layernorm.shape_0, layer.pre_feedforward_layernorm.shape_1)
    result.post_feedforward_layernorm = _tensor_from_device_ptr(base + post_ff_off, layer.post_feedforward_layernorm.shape_0, layer.post_feedforward_layernorm.shape_1)
    return result


def upload_expert_weights(
    mut stage: WeightStage, mut ctx: GPUContext, expert: MoEExpertWeights
) raises -> MoEExpertWeights:
    """Upload a single MoE expert's weights (gate/up/down_proj) to device staging.

    Only called for the 8 selected experts per token, not all 128.

    Args:
        stage: Reusable staging buffer.
        ctx: GPU context for the transfer.
        expert: Source expert weights with CPU/mmap pointers.

    Returns:
        A new MoEExpertWeights with device pointers.
    """
    stage.reset()

    var gate_off = stage._pack_tensor(expert.gate_proj)
    var up_off = stage._pack_tensor(expert.up_proj)
    var down_off = stage._pack_tensor(expert.down_proj)

    ctx.upload(stage.device_buf, stage.host_buf)

    var base = stage.device_buf.unsafe_ptr()
    var result = MoEExpertWeights()
    result.gate_proj = _tensor_from_device_ptr(base + gate_off, expert.gate_proj.shape_0, expert.gate_proj.shape_1)
    result.up_proj = _tensor_from_device_ptr(base + up_off, expert.up_proj.shape_0, expert.up_proj.shape_1)
    result.down_proj = _tensor_from_device_ptr(base + down_off, expert.down_proj.shape_0, expert.down_proj.shape_1)
    return result


def upload_vision_layer_weights(
    mut stage: WeightStage, mut ctx: GPUContext, layer: VisionLayerWeights
) raises -> VisionLayerWeights:
    """Upload a vision transformer layer's weights to device staging.

    Args:
        stage: Reusable staging buffer.
        ctx: GPU context for the transfer.
        layer: Source vision layer weights with CPU/mmap pointers.

    Returns:
        A new VisionLayerWeights with device pointers.
    """
    stage.reset()

    var q_off = stage._pack_tensor(layer.q_proj)
    var k_off = stage._pack_tensor(layer.k_proj)
    var v_off = stage._pack_tensor(layer.v_proj)
    var o_off = stage._pack_tensor(layer.o_proj)
    var fc1_off = stage._pack_tensor(layer.fc1)
    var fc2_off = stage._pack_tensor(layer.fc2)
    var ln1_off = stage._pack_tensor(layer.layer_norm1)
    var ln2_off = stage._pack_tensor(layer.layer_norm2)

    ctx.upload(stage.device_buf, stage.host_buf)

    var base = stage.device_buf.unsafe_ptr()
    var result = VisionLayerWeights()
    result.q_proj = _tensor_from_device_ptr(base + q_off, layer.q_proj.shape_0, layer.q_proj.shape_1)
    result.k_proj = _tensor_from_device_ptr(base + k_off, layer.k_proj.shape_0, layer.k_proj.shape_1)
    result.v_proj = _tensor_from_device_ptr(base + v_off, layer.v_proj.shape_0, layer.v_proj.shape_1)
    result.o_proj = _tensor_from_device_ptr(base + o_off, layer.o_proj.shape_0, layer.o_proj.shape_1)
    result.fc1 = _tensor_from_device_ptr(base + fc1_off, layer.fc1.shape_0, layer.fc1.shape_1)
    result.fc2 = _tensor_from_device_ptr(base + fc2_off, layer.fc2.shape_0, layer.fc2.shape_1)
    result.layer_norm1 = _tensor_from_device_ptr(base + ln1_off, layer.layer_norm1.shape_0, layer.layer_norm1.shape_1)
    result.layer_norm2 = _tensor_from_device_ptr(base + ln2_off, layer.layer_norm2.shape_0, layer.layer_norm2.shape_1)
    return result
