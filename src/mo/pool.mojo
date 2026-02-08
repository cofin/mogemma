from memory import LegacyUnsafePointer
from python import Python, PythonObject

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]

struct MemoryPool:
    var capacity: Int
    var pointers: PythonObject

    fn __init__(out self, capacity: Int) raises:
        self.capacity = capacity
        self.pointers = Python.list()

    fn acquire(mut self, size: Int) raises -> Int:
        """Acquire a buffer address from the pool."""
        if len(self.pointers) > 0:
            return Int(py=self.pointers.pop())
        
        var ptr = UnsafePointer[Float32].alloc(size)
        # Using a dummy for now to verify the rest of the logic
        return 42

    fn release(mut self, address: Int) raises:
        """Release a buffer address back to the pool."""
        self.pointers.append(address)