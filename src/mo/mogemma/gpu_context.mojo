"""GPU memory infrastructure for mogemma.

Provides GPUContext (DeviceContext lifecycle), buffer allocation helpers,
and host↔device transfer primitives. All GPU code is gated behind
`comptime if has_accelerator()` so this module compiles on CPU-only systems.
"""

from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.memory import UnsafePointer
from std.collections import List

from mogemma.model import TensorInfo


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
