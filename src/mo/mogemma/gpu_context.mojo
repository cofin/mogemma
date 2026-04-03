"""GPU memory infrastructure for mogemma.

Provides GPUContext (DeviceContext lifecycle), buffer allocation helpers,
and host↔device transfer primitives. All GPU code is gated behind
`comptime if has_accelerator()` so this module compiles on CPU-only systems.
"""

from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.memory import UnsafePointer
from std.collections import List

from mogemma.model import TensorInfo, LayerWeights, MoEExpertWeights, VisionLayerWeights


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
