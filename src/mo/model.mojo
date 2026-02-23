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

@fieldwise_init
struct ModelWeights(Movable):
    var embed_tokens: TensorInfo
    var norm: TensorInfo
    var layers: List[LayerWeights]
    
    fn __init__(out self):
        self.embed_tokens = TensorInfo(0, 0, 0)
        self.norm = TensorInfo(0, 0, 0)
        self.layers = List[LayerWeights]()

    @always_inline
    fn get_embedding(self, token_id: Int, out_ptr: UnsafePointer[Float32, MutExternalOrigin]):
        var hidden_size = self.embed_tokens.shape_1
        var src_ptr = self.embed_tokens.ptr + token_id * hidden_size
        
        # Simple copy loop
        for i in range(hidden_size):
            out_ptr.store(i, src_ptr.load(i))

