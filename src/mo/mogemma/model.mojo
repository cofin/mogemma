from std.memory import UnsafePointer
from std.math import cos, sin
from std.collections import List

# Model Weight Definitions for Gemma 4


@fieldwise_init
struct TensorInfo(Copyable, ImplicitlyCopyable, Movable):
    """Represents the metadata and memory pointer for a single model tensor."""

    var ptr: UnsafePointer[Float32, MutExternalOrigin]
    var scale_ptr: UnsafePointer[Float32, MutExternalOrigin]
    var i8_ptr: UnsafePointer[Int8, MutExternalOrigin]
    var is_quantized: Bool
    var shape_0: Int
    var shape_1: Int

    fn __init__(out self, p: Int, s0: Int, s1: Int):
        self.ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=p)
        self.scale_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=0)
        self.i8_ptr = UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=0)
        self.is_quantized = False
        self.shape_0 = s0
        self.shape_1 = s1

    fn __init__(out self):
        self.ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=0)
        self.scale_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=0)
        self.i8_ptr = UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=0)
        self.is_quantized = False
        self.shape_0 = 0
        self.shape_1 = 0

    fn __init__(out self, i8_p: Int, scale_p: Int, s0: Int, s1: Int):
        self.ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=0)
        self.scale_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=scale_p)
        self.i8_ptr = UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=i8_p)
        self.is_quantized = True
        self.shape_0 = s0
        self.shape_1 = s1


@fieldwise_init
struct LayerWeights(Copyable, ImplicitlyCopyable, Movable):
    """Container for all learnable parameter tensors within a single standard transformer layer."""

    var q_proj: TensorInfo
    var k_proj: TensorInfo
    var v_proj: TensorInfo
    var o_proj: TensorInfo
    var gate_proj: TensorInfo
    var up_proj: TensorInfo
    var down_proj: TensorInfo
    var input_layernorm: TensorInfo
    var post_attention_layernorm: TensorInfo
    var q_norm: TensorInfo
    var k_norm: TensorInfo
    var pre_feedforward_layernorm: TensorInfo
    var post_feedforward_layernorm: TensorInfo

    fn __init__(out self):
        self.q_proj = TensorInfo(0, 0, 0)
        self.k_proj = TensorInfo(0, 0, 0)
        self.v_proj = TensorInfo(0, 0, 0)
        self.o_proj = TensorInfo(0, 0, 0)
        self.gate_proj = TensorInfo(0, 0, 0)
        self.up_proj = TensorInfo(0, 0, 0)
        self.down_proj = TensorInfo(0, 0, 0)
        self.input_layernorm = TensorInfo(0, 0, 0)
        self.post_attention_layernorm = TensorInfo(0, 0, 0)
        self.q_norm = TensorInfo(0, 0, 0)
        self.k_norm = TensorInfo(0, 0, 0)
        self.pre_feedforward_layernorm = TensorInfo(0, 0, 0)
        self.post_feedforward_layernorm = TensorInfo(0, 0, 0)


@fieldwise_init
struct ModelWeights(Movable):
    """Top-level container for all standard Gemma model weights, including token embeddings and transformer layers."""

    var embed_tokens: TensorInfo
    var norm: TensorInfo
    var lm_head: TensorInfo
    var layers: List[LayerWeights]

    fn __init__(out self):
        self.embed_tokens = TensorInfo(0, 0, 0)
        self.norm = TensorInfo(0, 0, 0)
        self.lm_head = TensorInfo(0, 0, 0)
        self.layers = List[LayerWeights]()

    @always_inline
    fn get_embedding(self, token_id: Int, out_ptr: UnsafePointer[Float32, MutExternalOrigin]):
        var hidden_size = self.embed_tokens.shape_1
        var src_ptr = self.embed_tokens.ptr + token_id * hidden_size

        # Simple copy loop
        for i in range(hidden_size):
            out_ptr.store(i, src_ptr.load(i))


# Layer type constants for KVCache
alias LAYER_TYPE_SLIDING: UInt8 = 0
alias LAYER_TYPE_FULL: UInt8 = 1


struct KVCache(Movable):
    """KV cache for Gemma 4 hybrid sliding-window + full global attention.

    Sliding-window layers use a ring buffer of `window_size` slots.
    Full attention layers use a linear buffer of `max_context_len` slots.
    All storage is backed by a single contiguous arena per K and V.
    """

    var num_layers: Int
    var num_kv_heads: Int
    var head_dim: Int
    var window_size: Int
    var max_context_len: Int

    # Per-layer metadata
    var layer_types: List[UInt8]       # LAYER_TYPE_SLIDING or LAYER_TYPE_FULL per layer
    var layer_cache_sizes: List[Int]   # cache slot count per layer
    var layer_offsets: List[Int]       # byte offset (in Float32 elements) into arena per layer

    # Contiguous arena storage
    var k_cache: List[Float32]
    var v_cache: List[Float32]
    var k_ptr: UnsafePointer[Float32, MutExternalOrigin]
    var v_ptr: UnsafePointer[Float32, MutExternalOrigin]

    fn __init__(
        out self,
        num_layers: Int,
        num_kv_heads: Int,
        head_dim: Int,
        window_size: Int,
        max_context_len: Int,
        layer_types_ptr: UnsafePointer[UInt8, MutExternalOrigin],
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.max_context_len = max_context_len

        # Build per-layer metadata and compute offsets
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

        # Allocate contiguous arenas
        self.k_cache = List[Float32](length=total_elements, fill=0.0)
        self.v_cache = List[Float32](length=total_elements, fill=0.0)
        self.k_ptr = UnsafePointer[Float32, MutExternalOrigin](
            unsafe_from_address=Int(self.k_cache.unsafe_ptr())
        )
        self.v_ptr = UnsafePointer[Float32, MutExternalOrigin](
            unsafe_from_address=Int(self.v_cache.unsafe_ptr())
        )

    @always_inline
    fn write_kv(
        self,
        layer: Int,
        pos: Int,
        k_vec: UnsafePointer[Float32, MutExternalOrigin],
        v_vec: UnsafePointer[Float32, MutExternalOrigin],
    ):
        """Write K and V vectors for a single token into the cache for a given layer.

        For sliding layers, writes at `pos % window_size` (ring buffer).
        For full layers, writes at the linear position.
        """
        var kv_stride = self.num_kv_heads * self.head_dim
        var cache_size = self.layer_cache_sizes[layer]
        var write_pos: Int
        if self.layer_types[layer] == LAYER_TYPE_FULL:
            write_pos = pos
        else:
            write_pos = pos % cache_size  # ring buffer wrap

        var offset = self.layer_offsets[layer] + write_pos * kv_stride
        var k_dst = self.k_ptr + offset
        var v_dst = self.v_ptr + offset
        for i in range(kv_stride):
            k_dst.store(i, k_vec.load(i))
            v_dst.store(i, v_vec.load(i))

    @always_inline
    fn get_kv_ptrs(
        self, layer: Int
    ) -> (
        UnsafePointer[Float32, MutExternalOrigin],
        UnsafePointer[Float32, MutExternalOrigin],
    ):
        """Returns (k_ptr, v_ptr) pointing to the start of this layer's cache region."""
        var offset = self.layer_offsets[layer]
        return (self.k_ptr + offset, self.v_ptr + offset)

    @always_inline
    fn get_attention_range(self, layer: Int, pos: Int) -> (Int, Int):
        """Returns (valid_len, cache_size) for computing attention at the given position.

        For sliding layers: valid_len = min(pos + 1, window_size).
        For full layers: valid_len = pos + 1.
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
        return (valid_len, cache_size)

    fn reset(mut self):
        """Zero all cache buffers."""
        var total = len(self.k_cache)
        for i in range(total):
            self.k_cache[i] = 0.0
            self.v_cache[i] = 0.0

    fn total_elements(self) -> Int:
        """Returns total Float32 elements allocated across all layers (per K or V)."""
        if self.num_layers == 0:
            return 0
        var last = self.num_layers - 1
        var kv_stride = self.num_kv_heads * self.head_dim
        return self.layer_offsets[last] + self.layer_cache_sizes[last] * kv_stride


struct RoPETables(Movable):
    """Precomputed cos/sin frequency tables for dual-theta RoPE in Gemma 4.

    Sliding layers use theta=10,000 with full rotation over `window_size` positions.
    Full layers use theta=1,000,000 with partial rotation over `max_context_len` positions.
    The `rotary_dim` for full layers is `head_dim * partial_rotary_factor` (rounded to even).
    """

    var head_dim: Int
    var rotary_dim: Int  # dimensions that get RoPE on full layers (head_dim * partial_rotary_factor)

    # Sliding: [window_size, head_dim] — full rotation, theta=10K
    var sliding_cos: List[Float32]
    var sliding_sin: List[Float32]
    var sliding_cos_ptr: UnsafePointer[Float32, MutExternalOrigin]
    var sliding_sin_ptr: UnsafePointer[Float32, MutExternalOrigin]

    # Full: [max_context_len, rotary_dim] — partial rotation, theta=1M
    var full_cos: List[Float32]
    var full_sin: List[Float32]
    var full_cos_ptr: UnsafePointer[Float32, MutExternalOrigin]
    var full_sin_ptr: UnsafePointer[Float32, MutExternalOrigin]

    fn __init__(
        out self,
        head_dim: Int,
        partial_rotary_factor: Float32,
        window_size: Int,
        max_context_len: Int,
        theta_sliding: Float32 = 10000.0,
        theta_full: Float32 = 1000000.0,
    ):
        self.head_dim = head_dim
        # Round rotary_dim to nearest even number
        var raw_rotary = Int(Float32(head_dim) * partial_rotary_factor)
        self.rotary_dim = (raw_rotary // 2) * 2  # ensure even

        # --- Sliding tables: theta=10K, full head_dim rotation ---
        var sliding_half = head_dim // 2
        var sliding_len = window_size * sliding_half
        self.sliding_cos = List[Float32](length=sliding_len, fill=0.0)
        self.sliding_sin = List[Float32](length=sliding_len, fill=0.0)
        for t in range(window_size):
            for d in range(sliding_half):
                var exp = Float32(d * 2) / Float32(head_dim)
                var inv_freq = 1.0 / (theta_sliding ** exp)
                var freq = Float32(t) * inv_freq
                self.sliding_cos[t * sliding_half + d] = cos(freq)
                self.sliding_sin[t * sliding_half + d] = sin(freq)
        self.sliding_cos_ptr = UnsafePointer[Float32, MutExternalOrigin](
            unsafe_from_address=Int(self.sliding_cos.unsafe_ptr())
        )
        self.sliding_sin_ptr = UnsafePointer[Float32, MutExternalOrigin](
            unsafe_from_address=Int(self.sliding_sin.unsafe_ptr())
        )

        # --- Full tables: theta=1M, rotary_dim rotation ---
        var full_half = self.rotary_dim // 2
        var full_len = max_context_len * full_half
        self.full_cos = List[Float32](length=full_len, fill=0.0)
        self.full_sin = List[Float32](length=full_len, fill=0.0)
        for t in range(max_context_len):
            for d in range(full_half):
                var exp = Float32(d * 2) / Float32(self.rotary_dim)
                var inv_freq = 1.0 / (theta_full ** exp)
                var freq = Float32(t) * inv_freq
                self.full_cos[t * full_half + d] = cos(freq)
                self.full_sin[t * full_half + d] = sin(freq)
        self.full_cos_ptr = UnsafePointer[Float32, MutExternalOrigin](
            unsafe_from_address=Int(self.full_cos.unsafe_ptr())
        )
        self.full_sin_ptr = UnsafePointer[Float32, MutExternalOrigin](
            unsafe_from_address=Int(self.full_sin.unsafe_ptr())
        )

    @always_inline
    fn get_sliding_freqs(self, pos: Int) -> (
        UnsafePointer[Float32, MutExternalOrigin],
        UnsafePointer[Float32, MutExternalOrigin],
    ):
        """Returns (cos_ptr, sin_ptr) for sliding-layer RoPE at given position.

        Position is taken mod the table size for ring buffer compatibility.
        """
        var half = self.head_dim // 2
        var table_pos = pos % (len(self.sliding_cos) // half)
        var offset = table_pos * half
        return (self.sliding_cos_ptr + offset, self.sliding_sin_ptr + offset)

    @always_inline
    fn get_full_freqs(self, pos: Int) -> (
        UnsafePointer[Float32, MutExternalOrigin],
        UnsafePointer[Float32, MutExternalOrigin],
    ):
        """Returns (cos_ptr, sin_ptr) for full-layer RoPE at given position."""
        var half = self.rotary_dim // 2
        var offset = pos * half
        return (self.full_cos_ptr + offset, self.full_sin_ptr + offset)


