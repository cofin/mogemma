from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort
from std.memory import UnsafePointer, alloc
from std.math import cos, sin, sqrt
from std.collections import List

from mogemma.model import (
    ModelWeights,
    LayerWeights,
    VisionLayerWeights,
    VisionModelWeights,
    TensorInfo,
    KVCache,
    RoPETables,
    LAYER_TYPE_SLIDING,
    LAYER_TYPE_FULL,
)
from mogemma.layers import forward_gemma4_step, forward_gemma4_step_with_embedding, forward_vision_encoder
from mogemma.ops import rms_norm, vec_mat_mul


fn _ensure_step_logits(logits_obj: PythonObject, np: PythonObject) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var logits = np.asarray(logits_obj, dtype=np.float32)
    if Int(py=builtins.len(logits.shape)) != 1:
        raise Error("step output must be a 1D float32 tensor")
    if Int(py=logits.shape[0]) <= 0:
        raise Error("step output must contain at least one element")
    return logits


fn _ensure_embedding_matrix(
    embeddings_obj: PythonObject, expected_rows: Int, expected_cols: Int, np: PythonObject
) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var embeddings = np.asarray(embeddings_obj, dtype=np.float32)
    if Int(py=builtins.len(embeddings.shape)) != 2:
        raise Error("generate_embeddings output must be a 2D float32 matrix")
    if Int(py=embeddings.shape[0]) != expected_rows:
        raise Error("generate_embeddings output row count does not match inputs")
    if Int(py=embeddings.shape[1]) != expected_cols:
        raise Error("generate_embeddings output columns do not match hidden_size")
    return embeddings


def _tensor_from_meta(meta_obj: PythonObject, scale_obj: PythonObject) -> TensorInfo:
    try:
        var builtins = Python.import_module("builtins")
        if not builtins.bool(meta_obj):
            return TensorInfo(0, 0, 0)

        var meta_tuple = meta_obj
        var ptr_int = Int(py=meta_tuple[0])
        var shape_tuple = meta_tuple[1]

        var s0 = 0
        var s1 = 0
        if Int(py=builtins.len(shape_tuple)) > 0:
            s0 = Int(py=shape_tuple[0])
        if Int(py=builtins.len(shape_tuple)) > 1:
            s1 = Int(py=shape_tuple[1])

        if builtins.bool(scale_obj):
            var scale_tuple = scale_obj
            var scale_ptr_int = Int(py=scale_tuple[0])
            return TensorInfo(ptr_int, scale_ptr_int, s0, s1)

        return TensorInfo(ptr_int, s0, s1)
    except e:
        return TensorInfo(0, 0, 0)


@always_inline
fn _append_tensor(mut ptrs: List[Int], t: TensorInfo):
    if t.is_quantized:
        ptrs.append(Int(t.i8_ptr))
    else:
        ptrs.append(Int(t.ptr))
    ptrs.append(Int(t.scale_ptr))
    ptrs.append(t.shape_0)
    ptrs.append(t.shape_1)


@always_inline
fn _hydrate_tensor(ptr_array: UnsafePointer[Int, MutExternalOrigin], mut offset: Int) -> TensorInfo:
    var p = ptr_array[offset]
    var scale = ptr_array[offset + 1]
    var s0 = ptr_array[offset + 2]
    var s1 = ptr_array[offset + 3]
    offset += 4
    if scale == 0:
        return TensorInfo(p, s0, s1)
    else:
        return TensorInfo(p, scale, s0, s1)


def _get_tensor(metadata_obj: PythonObject, name: String) -> TensorInfo:
    try:
        var scale_name = name + "_scale"
        return _tensor_from_meta(metadata_obj.get(name), metadata_obj.get(scale_name))
    except e:
        return TensorInfo(0, 0, 0)


@always_inline
fn _step_scratch_len(hidden_size: Int, max_seq_len: Int, num_heads: Int) -> Int:
    return hidden_size * 160 + max_seq_len * num_heads * 2


@always_inline
fn _embedding_scratch_len(hidden_size: Int, max_seq_len: Int, num_heads: Int) -> Int:
    return hidden_size * 180 + max_seq_len * num_heads * 2


fn _allocate_transient_f32(length: Int) -> List[Float32]:
    var values = List[Float32](length=length, fill=0.0)
    return values^


@always_inline
fn _allocate_transient_i32(length: Int) -> List[Int32]:
    var values = List[Int32](length=length, fill=0)
    return values^


struct Appender:
    var list: List[Int]

    fn __init__(out self):
        self.list = List[Int]()

    fn append(mut self, t: TensorInfo):
        self.list.append(Int(t.ptr))
        self.list.append(Int(t.scale_ptr))
        self.list.append(t.shape_0)
        self.list.append(t.shape_1)

    fn append(mut self, val: Int):
        self.list.append(val)

    fn finish(mut self) -> List[Int]:
        var res = List[Int]()
        for i in range(len(self.list)):
            res.append(self.list[i])
        return res^


struct Hydrator:
    var ptr: UnsafePointer[Int, MutExternalOrigin]
    var offset: Int

    fn __init__(out self, ptr: UnsafePointer[Int, MutExternalOrigin]):
        self.ptr = ptr
        self.offset = 0

    fn next(mut self) -> TensorInfo:
        var t = TensorInfo()
        t.ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=self.ptr[self.offset])
        t.scale_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=self.ptr[self.offset + 1])
        t.shape_0 = self.ptr[self.offset + 2]
        t.shape_1 = self.ptr[self.offset + 3]
        self.offset += 4
        return t

    fn next_int(mut self) -> Int:
        var val = self.ptr[self.offset]
        self.offset += 1
        return val


comptime ArenaPtr = UnsafePointer[Float32, MutExternalOrigin]


struct MemoryArena:
    var ptr: ArenaPtr
    var size: Int

    fn __init__(out self, p: ArenaPtr, s: Int):
        self.ptr = p
        self.size = s

    fn __init__(out self, size: Int):
        var p = alloc[Float32](size)
        self.ptr = ArenaPtr(unsafe_from_address=Int(p))
        self.size = size
        # Zero the arena
        for i in range(size):
            self.ptr.store(i, 0.0)

    fn free(mut self):
        if Int(self.ptr) != 0:
            var p = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(self.ptr))
            p.free()
            self.ptr = ArenaPtr(unsafe_from_address=0)
            self.size = 0


# ── Weight Loading ──────────────────────────────────────────────────────────

fn _build_standard_runtime(metadata_obj: PythonObject) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var runtime = Python.dict()
    runtime["embed_tokens"] = metadata_obj.get("model.embed_tokens.weight")
    runtime["norm"] = metadata_obj.get("model.norm.weight")
    runtime["lm_head"] = metadata_obj.get("lm_head.weight")

    var layers = Python.list()
    var layer_idx = 0
    while True:
        var pfx = "model.layers." + String(layer_idx)
        var layernorm = metadata_obj.get(pfx + ".input_layernorm.weight")
        if not builtins.bool(layernorm):
            break

        var layer_entry = Python.list()
        layer_entry.append(layernorm)
        layer_entry.append(metadata_obj.get(pfx + ".post_attention_layernorm.weight"))
        layer_entry.append(metadata_obj.get(pfx + ".self_attn.q_proj.weight"))
        layer_entry.append(metadata_obj.get(pfx + ".self_attn.k_proj.weight"))
        layer_entry.append(metadata_obj.get(pfx + ".self_attn.v_proj.weight"))
        layer_entry.append(metadata_obj.get(pfx + ".self_attn.o_proj.weight"))
        layer_entry.append(metadata_obj.get(pfx + ".mlp.gate_proj.weight"))
        layer_entry.append(metadata_obj.get(pfx + ".mlp.up_proj.weight"))
        layer_entry.append(metadata_obj.get(pfx + ".mlp.down_proj.weight"))
        layer_entry.append(metadata_obj.get(pfx + ".self_attn.q_norm.weight"))
        layer_entry.append(metadata_obj.get(pfx + ".self_attn.k_norm.weight"))
        layer_entry.append(metadata_obj.get(pfx + ".pre_feedforward_layernorm.weight"))
        layer_entry.append(metadata_obj.get(pfx + ".post_feedforward_layernorm.weight"))
        # Scales
        layer_entry.append(metadata_obj.get(pfx + ".self_attn.q_proj.weight_scale"))
        layer_entry.append(metadata_obj.get(pfx + ".self_attn.k_proj.weight_scale"))
        layer_entry.append(metadata_obj.get(pfx + ".self_attn.v_proj.weight_scale"))
        layer_entry.append(metadata_obj.get(pfx + ".self_attn.o_proj.weight_scale"))
        layer_entry.append(metadata_obj.get(pfx + ".mlp.gate_proj.weight_scale"))
        layer_entry.append(metadata_obj.get(pfx + ".mlp.up_proj.weight_scale"))
        layer_entry.append(metadata_obj.get(pfx + ".mlp.down_proj.weight_scale"))
        layers.append(layer_entry)
        layer_idx += 1

    runtime["layers"] = layers
    return runtime


fn _build_model_from_runtime(runtime_obj: PythonObject) raises -> ModelWeights:
    var m = ModelWeights()

    m.embed_tokens = _tensor_from_meta(runtime_obj["embed_tokens"], PythonObject())
    m.norm = _tensor_from_meta(runtime_obj["norm"], PythonObject())
    m.lm_head = _tensor_from_meta(runtime_obj["lm_head"], PythonObject())

    var layers = runtime_obj["layers"]
    var num_layers = Int(py=Python.import_module("builtins").len(layers))
    for i in range(num_layers):
        var entry = layers[i]
        var layer = LayerWeights()
        layer.input_layernorm = _tensor_from_meta(entry[0], PythonObject())
        layer.post_attention_layernorm = _tensor_from_meta(entry[1], PythonObject())
        layer.q_proj = _tensor_from_meta(entry[2], entry[13])
        layer.k_proj = _tensor_from_meta(entry[3], entry[14])
        layer.v_proj = _tensor_from_meta(entry[4], entry[15])
        layer.o_proj = _tensor_from_meta(entry[5], entry[16])
        layer.gate_proj = _tensor_from_meta(entry[6], entry[17])
        layer.up_proj = _tensor_from_meta(entry[7], entry[18])
        layer.down_proj = _tensor_from_meta(entry[8], entry[19])
        layer.q_norm = _tensor_from_meta(entry[9], PythonObject())
        layer.k_norm = _tensor_from_meta(entry[10], PythonObject())
        layer.pre_feedforward_layernorm = _tensor_from_meta(entry[11], PythonObject())
        layer.post_feedforward_layernorm = _tensor_from_meta(entry[12], PythonObject())
        m.layers.append(layer^)

    return m^


fn _flatten_model_weights(m: ModelWeights) -> List[Int]:
    var appender = Appender()
    appender.append(m.embed_tokens)
    appender.append(m.norm)
    appender.append(m.lm_head)
    for i in range(len(m.layers)):
        var layer = m.layers[i]
        appender.append(layer.input_layernorm)
        appender.append(layer.post_attention_layernorm)
        appender.append(layer.q_proj)
        appender.append(layer.k_proj)
        appender.append(layer.v_proj)
        appender.append(layer.o_proj)
        appender.append(layer.gate_proj)
        appender.append(layer.up_proj)
        appender.append(layer.down_proj)
        appender.append(layer.q_norm)
        appender.append(layer.k_norm)
        appender.append(layer.pre_feedforward_layernorm)
        appender.append(layer.post_feedforward_layernorm)
    return appender.finish()


fn _hydrate_model_weights(ptr_array: UnsafePointer[Int, MutExternalOrigin], num_layers: Int) -> ModelWeights:
    var m = ModelWeights()
    var h = Hydrator(ptr_array)
    m.embed_tokens = h.next()
    m.norm = h.next()
    m.lm_head = h.next()
    for _ in range(num_layers):
        var layer = LayerWeights()
        layer.input_layernorm = h.next()
        layer.post_attention_layernorm = h.next()
        layer.q_proj = h.next()
        layer.k_proj = h.next()
        layer.v_proj = h.next()
        layer.o_proj = h.next()
        layer.gate_proj = h.next()
        layer.up_proj = h.next()
        layer.down_proj = h.next()
        layer.q_norm = h.next()
        layer.k_norm = h.next()
        layer.pre_feedforward_layernorm = h.next()
        layer.post_feedforward_layernorm = h.next()
        m.layers.append(layer^)
    return m^


# ── Vision Weight Loading ──────────────────────────────────────────────────

fn _build_vision_runtime(metadata_obj: PythonObject, num_vision_layers: Int) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var runtime = Python.dict()

    runtime["patch_embedding"] = metadata_obj.get("vision_tower.vision_model.embeddings.patch_embedding.weight")
    runtime["position_embedding"] = metadata_obj.get("vision_tower.vision_model.embeddings.position_embedding.weight")
    runtime["post_norm"] = metadata_obj.get("vision_tower.vision_model.post_layernorm.weight")
    runtime["projection"] = metadata_obj.get("multi_modal_projector.linear.weight")

    var layers = Python.list()
    for i in range(num_vision_layers):
        var pfx = "vision_tower.vision_model.encoder.layers." + String(i)
        var layer_entry = Python.list()
        layer_entry.append(metadata_obj.get(pfx + ".self_attn.q_proj.weight"))
        layer_entry.append(metadata_obj.get(pfx + ".self_attn.k_proj.weight"))
        layer_entry.append(metadata_obj.get(pfx + ".self_attn.v_proj.weight"))
        layer_entry.append(metadata_obj.get(pfx + ".self_attn.out_proj.weight"))
        layer_entry.append(metadata_obj.get(pfx + ".mlp.fc1.weight"))
        layer_entry.append(metadata_obj.get(pfx + ".mlp.fc2.weight"))
        layer_entry.append(metadata_obj.get(pfx + ".layer_norm1.weight"))
        layer_entry.append(metadata_obj.get(pfx + ".layer_norm2.weight"))
        layers.append(layer_entry)

    runtime["layers"] = layers
    return runtime


fn _build_vision_from_runtime(runtime_obj: PythonObject, num_vision_layers: Int) raises -> VisionModelWeights:
    var vm = VisionModelWeights()
    vm.patch_embedding = _tensor_from_meta(runtime_obj["patch_embedding"], PythonObject())
    vm.position_embedding = _tensor_from_meta(runtime_obj["position_embedding"], PythonObject())
    vm.post_norm = _tensor_from_meta(runtime_obj["post_norm"], PythonObject())
    vm.projection = _tensor_from_meta(runtime_obj["projection"], PythonObject())

    var layers = runtime_obj["layers"]
    for i in range(num_vision_layers):
        var entry = layers[i]
        var vl = VisionLayerWeights()
        vl.q_proj = _tensor_from_meta(entry[0], PythonObject())
        vl.k_proj = _tensor_from_meta(entry[1], PythonObject())
        vl.v_proj = _tensor_from_meta(entry[2], PythonObject())
        vl.o_proj = _tensor_from_meta(entry[3], PythonObject())
        vl.fc1 = _tensor_from_meta(entry[4], PythonObject())
        vl.fc2 = _tensor_from_meta(entry[5], PythonObject())
        vl.layer_norm1 = _tensor_from_meta(entry[6], PythonObject())
        vl.layer_norm2 = _tensor_from_meta(entry[7], PythonObject())
        vm.layers.append(vl^)

    return vm^


fn _flatten_vision_weights(vm: VisionModelWeights) -> List[Int]:
    var appender = Appender()
    appender.append(vm.patch_embedding)
    appender.append(vm.position_embedding)
    appender.append(vm.post_norm)
    appender.append(vm.projection)
    for i in range(len(vm.layers)):
        var layer = vm.layers[i]
        appender.append(layer.q_proj)
        appender.append(layer.k_proj)
        appender.append(layer.v_proj)
        appender.append(layer.o_proj)
        appender.append(layer.fc1)
        appender.append(layer.fc2)
        appender.append(layer.layer_norm1)
        appender.append(layer.layer_norm2)
    return appender.finish()


fn _hydrate_vision_weights(ptr_array: UnsafePointer[Int, MutExternalOrigin], num_vision_layers: Int) -> VisionModelWeights:
    var vm = VisionModelWeights()
    var h = Hydrator(ptr_array)
    vm.patch_embedding = h.next()
    vm.position_embedding = h.next()
    vm.post_norm = h.next()
    vm.projection = h.next()
    for _ in range(num_vision_layers):
        var vl = VisionLayerWeights()
        vl.q_proj = h.next()
        vl.k_proj = h.next()
        vl.v_proj = h.next()
        vl.o_proj = h.next()
        vl.fc1 = h.next()
        vl.fc2 = h.next()
        vl.layer_norm1 = h.next()
        vl.layer_norm2 = h.next()
        vm.layers.append(vl^)
    return vm^


# ── Gemma 4 Runtime Init ───────────────────────────────────────────────────

def _init_model_impl_mojo(
    metadata_obj: PythonObject,
    architecture_overrides_obj: PythonObject,
) raises -> PythonObject:
    """Initialize the Gemma 4 model runtime.

    Builds ModelWeights from tensor metadata, creates the hybrid KVCache and
    dual-theta RoPETables from architecture config, and allocates scratch memory.
    """
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")

    var py_dict = Python.dict()
    py_dict["engine"] = "Mojo Gemma 4 Inference Engine"
    py_dict["arch"] = "gemma4"

    # Build model weights from metadata
    var std_model = _build_standard_runtime(metadata_obj)
    var model_weights = _build_model_from_runtime(std_model)
    var num_layers = len(model_weights.layers)
    if num_layers == 0:
        raise Error("Invalid model weights: no layers found in metadata")

    # Infer architecture parameters from weight shapes
    var head_dim = model_weights.layers[0].q_norm.shape_0
    if head_dim == 0:
        head_dim = 256
    var num_heads = model_weights.layers[0].q_proj.shape_0 // head_dim
    var num_kv_heads = model_weights.layers[0].k_proj.shape_0 // head_dim
    var hidden_size = model_weights.embed_tokens.shape_1
    var intermediate_size = model_weights.layers[0].gate_proj.shape_0
    var vocab_size = model_weights.lm_head.shape_0

    # Flatten weight pointers for reconstruction in step_mojo
    var ptrs = _flatten_model_weights(model_weights)
    var ptrs_np = np.zeros(len(ptrs), dtype=np.uint64)
    for i in range(len(ptrs)):
        ptrs_np[i] = ptrs[i]
    py_dict["_tensor_pointers"] = ptrs_np

    # Read Gemma 4 architecture config from overrides
    var max_seq_len = 8192
    var window_size = 1024
    var partial_rotary_factor: Float32 = 0.5
    var k_eq_v = False
    var num_vision_layers = 0
    var vision_hidden_size = 0
    var vision_num_heads = 0
    var vision_intermediate_size = 0
    var image_token_id = 0

    if Int(py=builtins.len(architecture_overrides_obj)) > 0:
        if builtins.bool(architecture_overrides_obj.get("max_seq_len")):
            max_seq_len = Int(py=architecture_overrides_obj["max_seq_len"])
        if builtins.bool(architecture_overrides_obj.get("window_size")):
            window_size = Int(py=architecture_overrides_obj["window_size"])
        if builtins.bool(architecture_overrides_obj.get("partial_rotary_factor")):
            partial_rotary_factor = Float32(py=architecture_overrides_obj["partial_rotary_factor"])
        if builtins.bool(architecture_overrides_obj.get("k_eq_v")):
            k_eq_v = Int(py=architecture_overrides_obj["k_eq_v"]) != 0
        if builtins.bool(architecture_overrides_obj.get("num_vision_layers")):
            num_vision_layers = Int(py=architecture_overrides_obj["num_vision_layers"])
        if builtins.bool(architecture_overrides_obj.get("vision_hidden_size")):
            vision_hidden_size = Int(py=architecture_overrides_obj["vision_hidden_size"])
        if builtins.bool(architecture_overrides_obj.get("vision_num_heads")):
            vision_num_heads = Int(py=architecture_overrides_obj["vision_num_heads"])
        if builtins.bool(architecture_overrides_obj.get("vision_intermediate_size")):
            vision_intermediate_size = Int(py=architecture_overrides_obj["vision_intermediate_size"])
        if builtins.bool(architecture_overrides_obj.get("image_token_id")):
            image_token_id = Int(py=architecture_overrides_obj["image_token_id"])

    # Parse layer_types from overrides — passed as a Python list of ints (0=sliding, 1=full)
    var layer_types_list = List[UInt8](length=num_layers, fill=UInt8(LAYER_TYPE_SLIDING))
    if builtins.bool(architecture_overrides_obj.get("layer_types")):
        var lt_obj = architecture_overrides_obj["layer_types"]
        var lt_len = Int(py=builtins.len(lt_obj))
        if lt_len == num_layers:
            for i in range(num_layers):
                layer_types_list[i] = UInt8(Int(py=lt_obj[i]))

    # Create KVCache
    var layer_types_ptr = UnsafePointer[UInt8, MutExternalOrigin](
        unsafe_from_address=Int(layer_types_list.unsafe_ptr())
    )
    var kv_cache = KVCache(
        num_layers, num_kv_heads, head_dim, window_size, max_seq_len, layer_types_ptr,
    )

    # Create RoPETables
    var rope_tables = RoPETables(
        head_dim, partial_rotary_factor, window_size, max_seq_len,
    )

    # Store KVCache and RoPETables as heap-allocated objects referenced by pointer
    var kv_cache_ptr = alloc[KVCache](1)
    kv_cache_ptr.init_pointee_move(kv_cache^)
    py_dict["_kv_cache_ptr"] = Int(kv_cache_ptr)

    var rope_tables_ptr = alloc[RoPETables](1)
    rope_tables_ptr.init_pointee_move(rope_tables^)
    py_dict["_rope_tables_ptr"] = Int(rope_tables_ptr)

    _ = layer_types_list

    # Build vision weights if vision layers are present
    py_dict["num_vision_layers"] = num_vision_layers
    py_dict["vision_hidden_size"] = vision_hidden_size
    py_dict["vision_num_heads"] = vision_num_heads
    py_dict["vision_intermediate_size"] = vision_intermediate_size
    py_dict["image_token_id"] = image_token_id
    py_dict["_vision_weights_ptr"] = 0
    py_dict["vision_embeddings"] = Python.list()

    if num_vision_layers > 0:
        var vision_runtime = _build_vision_runtime(metadata_obj, num_vision_layers)
        var vision_weights = _build_vision_from_runtime(vision_runtime, num_vision_layers)

        var v_ptrs = _flatten_vision_weights(vision_weights)
        var v_ptrs_np = np.zeros(len(v_ptrs), dtype=np.uint64)
        for i in range(len(v_ptrs)):
            v_ptrs_np[i] = v_ptrs[i]
        py_dict["_vision_tensor_pointers"] = v_ptrs_np
        py_dict["vision_runtime"] = vision_runtime

        # Infer vision head_dim
        if vision_num_heads > 0 and vision_hidden_size > 0:
            py_dict["vision_head_dim"] = vision_hidden_size // vision_num_heads
        else:
            py_dict["vision_head_dim"] = 0

    # Allocate scratch memory
    var step_scratch_len = _step_scratch_len(hidden_size, max_seq_len, num_heads)
    var emb_scratch_len = _embedding_scratch_len(hidden_size, max_seq_len, num_heads)
    var max_scratch_len = step_scratch_len
    if emb_scratch_len > max_scratch_len:
        max_scratch_len = emb_scratch_len

    var emb_out_len = hidden_size
    var total_arena_len = max_scratch_len + emb_out_len

    var arena = MemoryArena(total_arena_len)
    var arena_base_ptr = arena.ptr

    var scratch_ptr = arena_base_ptr
    var emb_out_ptr = scratch_ptr + max_scratch_len

    py_dict["_arena_ptr"] = Int(arena_base_ptr)
    py_dict["_arena_size"] = total_arena_len
    py_dict["step_scratch"] = Int(scratch_ptr)
    py_dict["emb_out"] = Int(emb_out_ptr)

    # Store all architecture parameters for step_mojo to read
    py_dict["max_seq_len"] = max_seq_len
    py_dict["num_layers"] = num_layers
    py_dict["num_heads"] = num_heads
    py_dict["num_kv_heads"] = num_kv_heads
    py_dict["head_dim"] = head_dim
    py_dict["hidden_size"] = hidden_size
    py_dict["intermediate_size"] = intermediate_size
    py_dict["vocab_size"] = vocab_size
    py_dict["window_size"] = window_size
    py_dict["k_eq_v"] = 1 if k_eq_v else 0
    py_dict["step_scratch_len"] = step_scratch_len
    py_dict["embedding_scratch_len"] = emb_scratch_len
    py_dict["runtime"] = std_model
    py_dict["pos"] = 0

    return py_dict


# ── FFI Entry Points ───────────────────────────────────────────────────────

def init_model_mojo(metadata_obj: PythonObject) -> PythonObject:
    """Initializes the Gemma 4 inference engine from Python metadata."""
    try:
        var empty_overrides = Python.dict()
        return _init_model_impl_mojo(metadata_obj, empty_overrides)
    except e:
        abort(String("failed to initialize model: ", e))


fn init_model_with_options_mojo(
    metadata_obj: PythonObject,
    architecture_overrides_obj: PythonObject,
    device_selection_obj: PythonObject,
) raises -> PythonObject:
    """Initializes the Gemma 4 runtime with architecture overrides and device selection."""
    var llm = _init_model_impl_mojo(metadata_obj, architecture_overrides_obj)
    llm["device_selection"] = device_selection_obj
    llm["device_backend"] = device_selection_obj.get("backend")
    llm["device_kind"] = device_selection_obj.get("device_kind")
    llm["device_index"] = device_selection_obj.get("device_index")
    llm["device_request"] = device_selection_obj.get("requested")
    llm["device_availability_source"] = device_selection_obj.get("availability_source")
    llm["device_strict"] = device_selection_obj.get("strict")
    if Int(py=Python.import_module("builtins").len(architecture_overrides_obj)) > 0:
        llm["architecture_overrides"] = architecture_overrides_obj
    return llm


fn step_mojo(
    llm: PythonObject,
    token_id_obj: PythonObject,
    temp_obj: PythonObject,
    top_k_obj: PythonObject,
    top_p_obj: PythonObject,
) raises -> PythonObject:
    """Performs a single forward pass step for Gemma 4 autoregressive generation."""
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")

    var pos = Int(py=llm["pos"])
    var max_seq_len = Int(py=llm["max_seq_len"])

    if pos >= max_seq_len:
        raise Error("Sequence length exceeded")

    var token_id = Int(py=token_id_obj)

    # Hydrate model weights from flattened pointer array
    var num_layers = Int(py=llm["num_layers"])
    var tensor_pointers_obj = llm["_tensor_pointers"]
    var tensor_pointers_ptr = UnsafePointer[Int, MutExternalOrigin](
        unsafe_from_address=Int(py=tensor_pointers_obj.__array_interface__["data"][0])
    )
    var model = _hydrate_model_weights(tensor_pointers_ptr, num_layers)

    var hidden_size = Int(py=llm["hidden_size"])
    var vocab_size = Int(py=llm["vocab_size"])
    var head_dim = Int(py=llm["head_dim"])
    var num_heads = Int(py=llm["num_heads"])
    var num_kv_heads = Int(py=llm["num_kv_heads"])
    var intermediate_size = Int(py=llm["intermediate_size"])
    var k_eq_v = Int(py=llm["k_eq_v"]) != 0

    # Retrieve KVCache and RoPETables from heap pointers
    var kv_cache_ptr_int = Int(py=llm["_kv_cache_ptr"])
    var rope_tables_ptr_int = Int(py=llm["_rope_tables_ptr"])
    var kv_cache_ptr = UnsafePointer[KVCache, MutExternalOrigin](unsafe_from_address=kv_cache_ptr_int)
    var rope_tables_ptr = UnsafePointer[RoPETables, MutExternalOrigin](unsafe_from_address=rope_tables_ptr_int)

    var scratch_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(py=llm["step_scratch"]))

    var out_logits = np.zeros(vocab_size, dtype=np.float32)
    var out_logits_ptr = UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(py=out_logits.__array_interface__["data"][0])
    )

    forward_gemma4_step(
        out_logits_ptr,
        token_id,
        pos,
        model,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        kv_cache_ptr[],
        rope_tables_ptr[],
        k_eq_v,
        max_seq_len,
        scratch_ptr,
    )

    llm["pos"] = pos + 1

    return _ensure_step_logits(out_logits, np)


fn process_image_mojo(
    llm: PythonObject,
    patches_obj: PythonObject,
    grid_h_obj: PythonObject,
    grid_w_obj: PythonObject,
) raises -> PythonObject:
    """Process preprocessed image patches through the vision encoder.

    Stores resulting vision embeddings in llm['vision_embeddings'] list.
    Returns the number of output tokens (after pooling).
    """
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")

    var num_vision_layers = Int(py=llm["num_vision_layers"])
    if num_vision_layers == 0:
        raise Error("No vision layers configured")

    var vision_hidden_size = Int(py=llm["vision_hidden_size"])
    var vision_num_heads = Int(py=llm["vision_num_heads"])
    var vision_head_dim = Int(py=llm["vision_head_dim"])
    var vision_intermediate_size = Int(py=llm["vision_intermediate_size"])
    var hidden_size = Int(py=llm["hidden_size"])
    var grid_h = Int(py=grid_h_obj)
    var grid_w = Int(py=grid_w_obj)
    var num_patches = grid_h * grid_w

    # Read patches as float32 array [num_patches, patch_dim]
    var patches = np.asarray(patches_obj, dtype=np.float32)
    var patches_ptr = UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(py=patches.__array_interface__["data"][0])
    )

    # Hydrate vision weights
    var v_tensor_pointers_obj = llm["_vision_tensor_pointers"]
    var v_tensor_pointers_ptr = UnsafePointer[Int, MutExternalOrigin](
        unsafe_from_address=Int(py=v_tensor_pointers_obj.__array_interface__["data"][0])
    )
    var vision_weights = _hydrate_vision_weights(v_tensor_pointers_ptr, num_vision_layers)

    # Compute output token count after pooling
    var pool_kernel = 3
    var pooled_h = grid_h // pool_kernel
    var pooled_w = grid_w // pool_kernel
    var pooled_tokens = pooled_h * pooled_w

    # Allocate output: [pooled_tokens, hidden_size] (decoder dim)
    var out_np = np.zeros(Python.tuple(pooled_tokens, hidden_size), dtype=np.float32)
    var out_ptr = UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(py=out_np.__array_interface__["data"][0])
    )

    # Allocate vision scratch
    var vision_scratch_size = num_patches * vision_hidden_size * 60 + vision_num_heads * num_patches * num_patches * 4
    var vision_scratch = _allocate_transient_f32(vision_scratch_size)
    var vision_scratch_ptr = UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(vision_scratch.unsafe_ptr())
    )

    forward_vision_encoder(
        out_ptr, patches_ptr, vision_weights,
        num_patches, grid_h, grid_w,
        vision_hidden_size, vision_num_heads, vision_head_dim, vision_intermediate_size,
        hidden_size, vision_scratch_ptr,
    )

    _ = vision_scratch

    # Store vision embeddings as individual token vectors in the list
    var vision_embeddings = llm["vision_embeddings"]
    for t in range(pooled_tokens):
        var token_emb = np.zeros(hidden_size, dtype=np.float32)
        for d in range(hidden_size):
            _ = token_emb.__setitem__(d, value=out_np[t][d])
        vision_embeddings.append(token_emb)

    return PythonObject(pooled_tokens)


fn step_with_embedding_mojo(
    llm: PythonObject,
    embedding_obj: PythonObject,
    temp_obj: PythonObject,
    top_k_obj: PythonObject,
    top_p_obj: PythonObject,
) raises -> PythonObject:
    """Like step_mojo but uses a pre-computed embedding vector instead of token lookup.

    Used for injecting vision tokens during prefill.
    """
    var np = Python.import_module("numpy")

    var pos = Int(py=llm["pos"])
    var max_seq_len = Int(py=llm["max_seq_len"])
    if pos >= max_seq_len:
        raise Error("Sequence length exceeded")

    var embedding = np.asarray(embedding_obj, dtype=np.float32)
    var embedding_ptr = UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(py=embedding.__array_interface__["data"][0])
    )

    var num_layers = Int(py=llm["num_layers"])
    var tensor_pointers_obj = llm["_tensor_pointers"]
    var tensor_pointers_ptr = UnsafePointer[Int, MutExternalOrigin](
        unsafe_from_address=Int(py=tensor_pointers_obj.__array_interface__["data"][0])
    )
    var model = _hydrate_model_weights(tensor_pointers_ptr, num_layers)

    var hidden_size = Int(py=llm["hidden_size"])
    var vocab_size = Int(py=llm["vocab_size"])
    var head_dim = Int(py=llm["head_dim"])
    var num_heads = Int(py=llm["num_heads"])
    var num_kv_heads = Int(py=llm["num_kv_heads"])
    var intermediate_size = Int(py=llm["intermediate_size"])
    var k_eq_v = Int(py=llm["k_eq_v"]) != 0

    var kv_cache_ptr = UnsafePointer[KVCache, MutExternalOrigin](unsafe_from_address=Int(py=llm["_kv_cache_ptr"]))
    var rope_tables_ptr = UnsafePointer[RoPETables, MutExternalOrigin](unsafe_from_address=Int(py=llm["_rope_tables_ptr"]))
    var scratch_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(py=llm["step_scratch"]))

    var out_logits = np.zeros(vocab_size, dtype=np.float32)
    var out_logits_ptr = UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(py=out_logits.__array_interface__["data"][0])
    )

    forward_gemma4_step_with_embedding(
        out_logits_ptr,
        embedding_ptr,
        pos,
        model,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        kv_cache_ptr[],
        rope_tables_ptr[],
        k_eq_v,
        max_seq_len,
        scratch_ptr,
    )

    llm["pos"] = pos + 1

    return _ensure_step_logits(out_logits, np)


fn generate_embeddings_mojo(
    llm: PythonObject,
    input_array: PythonObject,
) raises -> PythonObject:
    """Generates mean-pooled embeddings by running tokens through the Gemma 4 forward pass."""
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")

    var batch_size = Int(py=builtins.len(input_array))
    if batch_size == 0:
        raise Error("inputs must contain at least one row")

    var seq_len = Int(py=builtins.len(input_array[0]))
    if seq_len == 0:
        raise Error("inputs must contain at least one token")

    # Read architecture params
    var num_layers = Int(py=llm["num_layers"])
    var hidden_size = Int(py=llm["hidden_size"])
    var head_dim = Int(py=llm["head_dim"])
    var num_heads = Int(py=llm["num_heads"])
    var num_kv_heads = Int(py=llm["num_kv_heads"])
    var intermediate_size = Int(py=llm["intermediate_size"])
    var vocab_size = Int(py=llm["vocab_size"])
    var max_seq_len = Int(py=llm["max_seq_len"])
    var k_eq_v = Int(py=llm["k_eq_v"]) != 0

    # Hydrate model weights
    var tensor_pointers_obj = llm["_tensor_pointers"]
    var tensor_pointers_ptr = UnsafePointer[Int, MutExternalOrigin](
        unsafe_from_address=Int(py=tensor_pointers_obj.__array_interface__["data"][0])
    )
    var model = _hydrate_model_weights(tensor_pointers_ptr, num_layers)

    # Retrieve KVCache and RoPETables
    var kv_cache_ptr_int = Int(py=llm["_kv_cache_ptr"])
    var rope_tables_ptr_int = Int(py=llm["_rope_tables_ptr"])
    var kv_cache_ptr = UnsafePointer[KVCache, MutExternalOrigin](unsafe_from_address=kv_cache_ptr_int)
    var rope_tables_ptr = UnsafePointer[RoPETables, MutExternalOrigin](unsafe_from_address=rope_tables_ptr_int)

    # Allocate scratch and output accumulator
    var scratch_len = _embedding_scratch_len(hidden_size, max_seq_len, num_heads)
    var scratch_local = _allocate_transient_f32(scratch_len)
    var scratch_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(scratch_local.unsafe_ptr()))

    # Output: [batch_size, hidden_size]
    var result_np = np.zeros(Python.tuple(batch_size, hidden_size), dtype=np.float32)

    # For each batch item, run forward pass for each token and mean-pool
    for b in range(batch_size):
        var seq_list = input_array[b]
        var actual_seq_len = Int(py=builtins.len(seq_list))

        # Reset KV cache for each sequence
        kv_cache_ptr[].reset()

        # Accumulator for mean pooling
        var emb_acc = _allocate_transient_f32(hidden_size)
        var emb_acc_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(emb_acc.unsafe_ptr()))

        # Process each token
        var out_logits = _allocate_transient_f32(vocab_size)
        var out_logits_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(out_logits.unsafe_ptr()))

        for t in range(actual_seq_len):
            var token_id = Int(py=seq_list[t])

            forward_gemma4_step(
                out_logits_ptr,
                token_id,
                t,
                model,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                intermediate_size,
                vocab_size,
                kv_cache_ptr[],
                rope_tables_ptr[],
                k_eq_v,
                max_seq_len,
                scratch_ptr,
            )

            # The hidden state before LM head projection is in scratch at position 1 (next_state)
            # Actually forward_gemma4_step writes logits, but we need the hidden state for embeddings.
            # For mean-pooling, we use the norm output which is at scratch_ptr + hidden_size
            # (next_state = scratch_ptr + hidden_size, which has the RMSNorm'd output after final layer)
            var norm_out_ptr = scratch_ptr + hidden_size
            for i in range(hidden_size):
                emb_acc_ptr.store(i, emb_acc_ptr.load(i) + norm_out_ptr.load(i))

        # Mean pool
        var scale = 1.0 / Float32(actual_seq_len)
        for i in range(hidden_size):
            _ = result_np.__setitem__(Python.tuple(b, i), value=emb_acc_ptr.load(i) * scale)

        _ = emb_acc
        _ = out_logits

    _ = scratch_local

    return _ensure_embedding_matrix(result_np, batch_size, hidden_size, np)


def _free_arena_impl_mojo(llm: PythonObject) raises:
    var builtins = Python.import_module("builtins")

    # Free the scratch arena
    var ptr_addr = Int(py=builtins.getattr(llm, "get")("_arena_ptr", 0))
    if ptr_addr != 0:
        var ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=ptr_addr)
        ptr.free()
        llm["_arena_ptr"] = 0
        llm["_arena_size"] = 0

    # Free KVCache
    var kv_ptr_addr = Int(py=builtins.getattr(llm, "get")("_kv_cache_ptr", 0))
    if kv_ptr_addr != 0:
        var kv_ptr = UnsafePointer[KVCache, MutExternalOrigin](unsafe_from_address=kv_ptr_addr)
        kv_ptr.destroy_pointee()
        kv_ptr.free()
        llm["_kv_cache_ptr"] = 0

    # Free RoPETables
    var rope_ptr_addr = Int(py=builtins.getattr(llm, "get")("_rope_tables_ptr", 0))
    if rope_ptr_addr != 0:
        var rope_ptr = UnsafePointer[RoPETables, MutExternalOrigin](unsafe_from_address=rope_ptr_addr)
        rope_ptr.destroy_pointee()
        rope_ptr.free()
        llm["_rope_tables_ptr"] = 0


def free_arena_mojo(llm: PythonObject) raises:
    """Explicitly frees the memory arena and Gemma 4 runtime structures."""
    _free_arena_impl_mojo(llm)


def reset_cache_mojo(llm: PythonObject) raises:
    """Zeros the hybrid KV cache buffers."""
    var kv_cache_ptr_int = Int(py=llm["_kv_cache_ptr"])
    if kv_cache_ptr_int != 0:
        var kv_cache_ptr = UnsafePointer[KVCache, MutExternalOrigin](unsafe_from_address=kv_cache_ptr_int)
        kv_cache_ptr[].reset()


fn test_ffi_mojo(
    llm: PythonObject,
    token_id_obj: PythonObject,
    temp_obj: PythonObject,
    top_k_obj: PythonObject,
    top_p_obj: PythonObject,
) raises -> PythonObject:
    return step_mojo(llm, token_id_obj, temp_obj, top_k_obj, top_p_obj)


@export
fn PyInit__core() -> PythonObject:
    try:
        var b = PythonModuleBuilder("_core")
        b.def_function[init_model_mojo]("init_model")
        b.def_function[init_model_with_options_mojo]("init_model_with_options")
        b.def_function[generate_embeddings_mojo]("generate_embeddings")
        b.def_function[step_mojo]("step")
        b.def_function[free_arena_mojo]("free_arena")
        b.def_function[reset_cache_mojo]("reset_cache")
        b.def_function[test_ffi_mojo]("test_ffi")
        b.def_function[process_image_mojo]("process_image")
        b.def_function[step_with_embedding_mojo]("step_with_embedding")
        return b.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
