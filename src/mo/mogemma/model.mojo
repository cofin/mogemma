from memory import UnsafePointer
from collections import List

# Model Weight Definitions for Gemma 3

@fieldwise_init
struct TensorInfo(Copyable, Movable):
    var ptr: UnsafePointer[Float32, MutExternalOrigin]
    var shape_0: Int
    var shape_1: Int

    fn __init__(out self, p: Int, s0: Int, s1: Int):
        self.ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=p)
        self.shape_0 = s0
        self.shape_1 = s1

@fieldwise_init
struct LayerWeights(Copyable, Movable):
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


struct KVCache(Movable):
    var max_seq_len: Int
    var num_layers: Int
    var num_kv_heads: Int
    var head_dim: Int
    
    var k_cache: List[Float32]
    var v_cache: List[Float32]
    
    var k_ptr: UnsafePointer[Float32, MutExternalOrigin]
    var v_ptr: UnsafePointer[Float32, MutExternalOrigin]
    
    fn __init__(out self, max_seq_len: Int, num_layers: Int, num_kv_heads: Int, head_dim: Int):
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        var size = num_layers * max_seq_len * num_kv_heads * head_dim
        self.k_cache = List[Float32](length=size, fill=0.0)
        self.v_cache = List[Float32](length=size, fill=0.0)
        
        self.k_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(self.k_cache.unsafe_ptr()))
        self.v_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(self.v_cache.unsafe_ptr()))



@fieldwise_init
struct AltUpWeights(Copyable, Movable):
    var router: TensorInfo
    var router_norm: TensorInfo
    var prediction_coefs: TensorInfo
    var correction_coefs: TensorInfo
    var output_scale: TensorInfo

    fn __init__(out self):
        self.router = TensorInfo(0, 0, 0)
        self.router_norm = TensorInfo(0, 0, 0)
        self.prediction_coefs = TensorInfo(0, 0, 0)
        self.correction_coefs = TensorInfo(0, 0, 0)
        self.output_scale = TensorInfo(0, 0, 0)

@fieldwise_init
struct LaurelWeights(Copyable, Movable):
    var down_proj: TensorInfo
    var up_proj: TensorInfo
    var norm: TensorInfo

    fn __init__(out self):
        self.down_proj = TensorInfo(0, 0, 0)
        self.up_proj = TensorInfo(0, 0, 0)
        self.norm = TensorInfo(0, 0, 0)

@fieldwise_init
struct PerLayerMapWeights(Copyable, Movable):
    var gate: TensorInfo
    var projection: TensorInfo
    var norm: TensorInfo

    fn __init__(out self):
        self.gate = TensorInfo(0, 0, 0)
        self.projection = TensorInfo(0, 0, 0)
        self.norm = TensorInfo(0, 0, 0)

@fieldwise_init
struct NanoLayerWeights(Movable, Copyable):
    var base: LayerWeights
    var altup: AltUpWeights
    var laurel: LaurelWeights
    var per_layer_map: PerLayerMapWeights

    fn __init__(out self):
        self.base = LayerWeights()
        self.altup = AltUpWeights()
        self.laurel = LaurelWeights()
        self.per_layer_map = PerLayerMapWeights()

@fieldwise_init
struct NanoModelWeights(Movable):
    var embed_tokens: TensorInfo
    var norm: TensorInfo
    var lm_head: TensorInfo
    var per_layer_embed: TensorInfo
    var per_layer_projection: TensorInfo
    var per_layer_norm: TensorInfo
    var altup_projections: List[TensorInfo]
    var altup_unembeds: List[TensorInfo]
    var layers: List[NanoLayerWeights]

    fn __init__(out self):
        self.embed_tokens = TensorInfo(0, 0, 0)
        self.norm = TensorInfo(0, 0, 0)
        self.lm_head = TensorInfo(0, 0, 0)
        self.per_layer_embed = TensorInfo(0, 0, 0)
        self.per_layer_projection = TensorInfo(0, 0, 0)
        self.per_layer_norm = TensorInfo(0, 0, 0)
        self.altup_projections = List[TensorInfo]()
        self.altup_unembeds = List[TensorInfo]()
        self.layers = List[NanoLayerWeights]()
        
    @always_inline
    fn get_embedding(self, token_id: Int, out_ptr: UnsafePointer[Float32, MutExternalOrigin]):
        var hidden_size = self.embed_tokens.shape_1
        var src_ptr = self.embed_tokens.ptr + token_id * hidden_size
        
        # Simple copy loop
        for i in range(hidden_size):
            out_ptr.store(i, src_ptr.load(i))

