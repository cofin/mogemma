from python import Python, PythonObject
from python.bindings import PythonModuleBuilder
from os import abort
from memory import UnsafePointer
from math import cos, sin
from collections import List

from model import ModelWeights, LayerWeights, TensorInfo, KVCache
from layers import forward_sequence, forward_step

fn _ensure_step_logits(logits_obj: PythonObject, np: PythonObject) raises -> PythonObject:
    var logits = np.asarray(logits_obj, dtype=np.float32)
    var builtins = Python.import_module("builtins")
    if Int(py=builtins.len(logits.shape)) != 1:
        raise Error("step output must be a 1D float32 tensor")
    if Int(py=logits.shape[0]) <= 0:
        raise Error("step output must contain at least one element")
    return logits


fn _ensure_embedding_matrix(embeddings_obj: PythonObject, expected_rows: Int, np: PythonObject) raises -> PythonObject:
    var embeddings = np.asarray(embeddings_obj, dtype=np.float32)
    var builtins = Python.import_module("builtins")
    if Int(py=builtins.len(embeddings.shape)) != 2:
        raise Error("generate_embeddings output must be a 2D float32 matrix")
    if Int(py=embeddings.shape[0]) != expected_rows:
        raise Error("generate_embeddings output row count does not match inputs")
    if Int(py=embeddings.shape[1]) != 768:
        raise Error("generate_embeddings output must have 768 columns")
    return embeddings


fn _get_tensor(metadata_obj: PythonObject, name: String) raises -> TensorInfo:
    var builtins = Python.import_module("builtins")
    if not builtins.bool(metadata_obj.get(name)):
        return TensorInfo(0, 0, 0)
        
    var meta_tuple = metadata_obj[name]
    var ptr_int = Int(py=meta_tuple[0])
    var shape_tuple = meta_tuple[1]
    
    var s0 = 0
    var s1 = 0
    if Int(py=builtins.len(shape_tuple)) > 0:
        s0 = Int(py=shape_tuple[0])
    if Int(py=builtins.len(shape_tuple)) > 1:
        s1 = Int(py=shape_tuple[1])
        
    return TensorInfo(ptr_int, s0, s1)

fn _build_model(metadata_obj: PythonObject) raises -> ModelWeights:
    var builtins = Python.import_module("builtins")
    var m = ModelWeights()
    
    m.embed_tokens = _get_tensor(metadata_obj, "model.embed_tokens.weight")
    m.norm = _get_tensor(metadata_obj, "model.norm.weight")
    m.lm_head = _get_tensor(metadata_obj, "lm_head.weight")
    
    var num_layers = 0
    while True:
        var k = "model.layers." + String(num_layers) + ".input_layernorm.weight"
        var t = _get_tensor(metadata_obj, k)
        if t.ptr == UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=0):
            break
            
        var layer = LayerWeights()
        layer.input_layernorm = t^
        layer.post_attention_layernorm = _get_tensor(metadata_obj, "model.layers." + String(num_layers) + ".post_attention_layernorm.weight")
        layer.q_proj = _get_tensor(metadata_obj, "model.layers." + String(num_layers) + ".self_attn.q_proj.weight")
        layer.k_proj = _get_tensor(metadata_obj, "model.layers." + String(num_layers) + ".self_attn.k_proj.weight")
        layer.v_proj = _get_tensor(metadata_obj, "model.layers." + String(num_layers) + ".self_attn.v_proj.weight")
        layer.o_proj = _get_tensor(metadata_obj, "model.layers." + String(num_layers) + ".self_attn.o_proj.weight")
        layer.gate_proj = _get_tensor(metadata_obj, "model.layers." + String(num_layers) + ".mlp.gate_proj.weight")
        layer.up_proj = _get_tensor(metadata_obj, "model.layers." + String(num_layers) + ".mlp.up_proj.weight")
        layer.down_proj = _get_tensor(metadata_obj, "model.layers." + String(num_layers) + ".mlp.down_proj.weight")
        
        m.layers.append(layer^)
        num_layers += 1
        
    return m^

fn init_model_mojo(
    metadata_obj: PythonObject
) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var np = Python.import_module("numpy")
    
    var py_dict = Python.dict()
    py_dict["engine"] = "Mojo Pure Inference Engine"
    py_dict["metadata"] = metadata_obj
    
    # Pre-allocate cache arrays and keep them alive in the dict as numpy arrays
    var model_weights = _build_model(metadata_obj)
    var num_layers = len(model_weights.layers)
    
    if num_layers == 0:
        raise Error("Invalid model weights: no layers found in metadata")
        
    var head_dim = 256
    var num_kv_heads = model_weights.layers[0].k_proj.shape_0 // head_dim
    var max_seq_len = 8192 # default max seq len
    
    var size = num_layers * max_seq_len * num_kv_heads * head_dim
    var k_cache = np.zeros(size, dtype=np.float32)
    var v_cache = np.zeros(size, dtype=np.float32)
    
    var freqs_cos = np.zeros(max_seq_len * head_dim, dtype=np.float32)
    var freqs_sin = np.zeros(max_seq_len * head_dim, dtype=np.float32)
    
    var freqs_cos_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(py=freqs_cos.__array_interface__["data"][0]))
    var freqs_sin_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(py=freqs_sin.__array_interface__["data"][0]))
    
    # RoPE precompute
    var base: Float32 = 10000.0
    for t in range(max_seq_len):
        for d in range(head_dim // 2):
            var exp = Float32(d * 2) / Float32(head_dim)
            var inv_freq = 1.0 / (base ** exp)
            var freq = Float32(t) * inv_freq
            freqs_cos_ptr.store(t * head_dim + d, cos(freq))
            freqs_sin_ptr.store(t * head_dim + d, sin(freq))
            
    py_dict["k_cache"] = k_cache
    py_dict["v_cache"] = v_cache
    py_dict["freqs_cos"] = freqs_cos
    py_dict["freqs_sin"] = freqs_sin
    py_dict["max_seq_len"] = max_seq_len
    py_dict["pos"] = 0
    return py_dict

fn step_mojo(
    llm: PythonObject,
    token_id_obj: PythonObject,
    temp_obj: PythonObject,
    top_k_obj: PythonObject,
    top_p_obj: PythonObject,
) raises -> PythonObject:
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")
    
    var pos = Int(py=llm["pos"])
    var max_seq_len = Int(py=llm["max_seq_len"])
    
    if pos >= max_seq_len:
        raise Error("Sequence length exceeded")
        
    var token_id = Int(py=token_id_obj)
    
    var metadata_obj = llm["metadata"]
    var model_weights = _build_model(metadata_obj)
    var num_layers = len(model_weights.layers)
    
    if num_layers == 0:
        raise Error("Invalid model weights: no layers found in metadata")
    
    var hidden_size = model_weights.embed_tokens.shape_1
    var vocab_size = model_weights.lm_head.shape_0
    var head_dim = 256 # Assume 256 for Gemma 3
    var num_heads = model_weights.layers[0].q_proj.shape_0 // head_dim
    var num_kv_heads = model_weights.layers[0].k_proj.shape_0 // head_dim
    var intermediate_size = model_weights.layers[0].gate_proj.shape_0
    
    var k_cache = llm["k_cache"]
    var v_cache = llm["v_cache"]
    var freqs_cos = llm["freqs_cos"]
    var freqs_sin = llm["freqs_sin"]
    
    var kv_cache_k_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(py=k_cache.__array_interface__["data"][0]))
    var kv_cache_v_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(py=v_cache.__array_interface__["data"][0]))
    var freqs_cos_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(py=freqs_cos.__array_interface__["data"][0]))
    var freqs_sin_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(py=freqs_sin.__array_interface__["data"][0]))
    
    var scratch = List[Float32](length=hidden_size * 50, fill=0.0) # generous scratch space
    var scratch_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(scratch.unsafe_ptr()))
    
    var out_logits = np.zeros(vocab_size, dtype=np.float32)
    var out_logits_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(py=out_logits.__array_interface__["data"][0]))
    
    forward_step(
        out_logits_ptr,
        token_id,
        pos,
        model_weights,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        freqs_cos_ptr,
        freqs_sin_ptr,
        kv_cache_k_ptr,
        kv_cache_v_ptr,
        max_seq_len,
        scratch_ptr
    )
    
    _ = scratch[0]
    llm["pos"] = pos + 1
    
    return _ensure_step_logits(out_logits, np)

fn generate_embeddings_mojo(
    llm: PythonObject,
    input_array: PythonObject,
) raises -> PythonObject:
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")

    var np_input = np.asarray(input_array, dtype=np.int32)
    var batch_size = Int(py=np_input.shape[0])
    var seq_len = Int(py=np_input.shape[1])
    
    var metadata_obj = llm["metadata"]
    var model_weights = _build_model(metadata_obj)
    var num_layers = len(model_weights.layers)
    
    if num_layers == 0:
        raise Error("Invalid model weights: no layers found in metadata")
    
    var hidden_size = model_weights.embed_tokens.shape_1
    var head_dim = 256 # Assume 256 for Gemma 3
    var num_heads = model_weights.layers[0].q_proj.shape_0 // head_dim
    var num_kv_heads = model_weights.layers[0].k_proj.shape_0 // head_dim
    var intermediate_size = model_weights.layers[0].gate_proj.shape_0
    
    var max_seq_len = seq_len
    
    # RoPE precompute
    var freqs_cos = List[Float32](length=max_seq_len * head_dim, fill=0.0)
    var freqs_sin = List[Float32](length=max_seq_len * head_dim, fill=0.0)
    var base: Float32 = 10000.0
    for t in range(max_seq_len):
        for d in range(head_dim // 2):
            var exp = Float32(d * 2) / Float32(head_dim)
            var inv_freq = 1.0 / (base ** exp)
            var freq = Float32(t) * inv_freq
            freqs_cos[t * head_dim + d] = cos(freq)
            freqs_sin[t * head_dim + d] = sin(freq)
            
    # Allocations for intermediate state
    var kv_cache_k = List[Float32](length=num_layers * max_seq_len * num_kv_heads * head_dim, fill=0.0)
    var kv_cache_v = List[Float32](length=num_layers * max_seq_len * num_kv_heads * head_dim, fill=0.0)
    var scratch = List[Float32](length=hidden_size * 50, fill=0.0) # generous scratch space
    var emb_out = List[Float32](length=batch_size * hidden_size, fill=0.0)
    var input_ids = List[Int32](length=seq_len, fill=0)

    # Convert lists to pointers
    var freqs_cos_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(freqs_cos.unsafe_ptr()))
    var freqs_sin_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(freqs_sin.unsafe_ptr()))
    var kv_cache_k_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(kv_cache_k.unsafe_ptr()))
    var kv_cache_v_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(kv_cache_v.unsafe_ptr()))
    var scratch_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(scratch.unsafe_ptr()))
    var emb_out_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(emb_out.unsafe_ptr()))
    var input_ids_ptr = UnsafePointer[Int32, MutExternalOrigin](unsafe_from_address=Int(input_ids.unsafe_ptr()))
    
    # Process each sequence in the batch
    for b in range(batch_size):
        # Extract input_ids for this batch
        for t in range(seq_len):
            var token_py = np_input[b][t]
            input_ids[t] = Int32(Int(py=token_py))
            
        var seq_out_ptr = emb_out_ptr + b * hidden_size
        
        forward_sequence(
            seq_out_ptr,
            input_ids_ptr,
            seq_len,
            model_weights,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            freqs_cos_ptr,
            freqs_sin_ptr,
            kv_cache_k_ptr,
            kv_cache_v_ptr,
            max_seq_len,
            scratch_ptr
        )
        
    # Return as numpy array
    var result_np = np.zeros(Python.tuple(batch_size, hidden_size), dtype=np.float32)
    # Copy from emb_out back to numpy
    for b in range(batch_size):
        for i in range(hidden_size):
            var val = emb_out[b * hidden_size + i]
            result_np[b][i] = val
            
    # We must ensure refs to Mojo lists are kept alive till here.
    _ = freqs_cos[0]
    _ = freqs_sin[0]
    _ = kv_cache_k[0]
    _ = kv_cache_v[0]
    _ = scratch[0]
    _ = emb_out[0]
    _ = input_ids[0]
    
    return _ensure_embedding_matrix(result_np, batch_size, np)

@export
fn PyInit__core() -> PythonObject:
    try:
        var b = PythonModuleBuilder("_core")
        b.def_function[init_model_mojo]("init_model")
        b.def_function[generate_embeddings_mojo]("generate_embeddings")
        b.def_function[step_mojo]("step")
        return b.finalize()
    except e:
        abort(String("failed to create Python module: ", e))


