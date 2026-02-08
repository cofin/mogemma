from memory import LegacyUnsafePointer
from sys import size_of

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]

@fieldwise_init
struct KVCache(ImplicitlyCopyable):
    """Stateful KV cache for Gemma 3 inference."""
    var num_layers: Int
    var batch_size: Int
    var max_seq_len: Int
    var head_dim: Int
    var num_heads: Int
    var k_ptr: UnsafePointer[Float32]
    var v_ptr: UnsafePointer[Float32]

    fn __init__(out self, num_layers: Int, batch_size: Int, max_seq_len: Int, head_dim: Int, num_heads: Int):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.num_heads = num_heads
        
        var total_elts = num_layers * batch_size * num_heads * max_seq_len * head_dim
        self.k_ptr = UnsafePointer[Float32].alloc(total_elts)
        self.v_ptr = UnsafePointer[Float32].alloc(total_elts)
        
        # Initialize with zeros
        for i in range(total_elts):
            self.k_ptr[i] = 0.0
            self.v_ptr[i] = 0.0

    fn is_allocated(self) -> Bool:
        return True

    fn get_k(self, layer: Int, batch: Int, head: Int, seq: Int, dim: Int) -> Float32:
        var idx = layer * (self.batch_size * self.num_heads * self.max_seq_len * self.head_dim) + \
                  batch * (self.num_heads * self.max_seq_len * self.head_dim) + \
                  head * (self.max_seq_len * self.head_dim) + \
                  seq * (self.head_dim) + \
                  dim
        return self.k_ptr[idx]

    fn set_k(mut self, layer: Int, batch: Int, head: Int, seq: Int, dim: Int, val: Float32):
        var idx = layer * (self.batch_size * self.num_heads * self.max_seq_len * self.head_dim) + \
                  batch * (self.num_heads * self.max_seq_len * self.head_dim) + \
                  head * (self.max_seq_len * self.head_dim) + \
                  seq * (self.head_dim) + \
                  dim
        self.k_ptr[idx] = val