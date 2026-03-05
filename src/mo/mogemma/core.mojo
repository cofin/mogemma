from python import Python, PythonObject
from python.bindings import PythonModuleBuilder
from os import abort
from memory import UnsafePointer
from math import cos, sin
from collections import List

from mogemma.model import (
    ModelWeights,
    LayerWeights,
    TensorInfo,
    KVCache,
    NanoModelWeights,
    NanoLayerWeights,
    AltUpWeights,
    LaurelWeights,
    PerLayerMapWeights,
)
from mogemma.layers import (
    forward_sequence,
    forward_step,
    forward_nano_sequence,
    forward_nano_step,
)

fn _detect_architecture(metadata_obj: PythonObject) raises -> String:
    var builtins = Python.import_module("builtins")
    # Gemma 3 Nano has AltUp router weights
    if builtins.bool(metadata_obj.__contains__("model.layers.0.altup.router.weight")):
        return "nano"
    return "standard"

fn _ensure_step_logits(logits_obj: PythonObject, np: PythonObject) raises -> PythonObject:
    var logits = np.asarray(logits_obj, dtype=np.float32)
    var builtins = Python.import_module("builtins")
    if Int(py=builtins.len(logits.shape)) != 1:
        raise Error("step output must be a 1D float32 tensor")
    if Int(py=logits.shape[0]) <= 0:
        raise Error("step output must contain at least one element")
    return logits


fn _ensure_embedding_matrix(embeddings_obj: PythonObject, expected_rows: Int, expected_cols: Int, np: PythonObject) raises -> PythonObject:
    var embeddings = np.asarray(embeddings_obj, dtype=np.float32)
    var builtins = Python.import_module("builtins")
    if Int(py=builtins.len(embeddings.shape)) != 2:
        raise Error("generate_embeddings output must be a 2D float32 matrix")
    if Int(py=embeddings.shape[0]) != expected_rows:
        raise Error("generate_embeddings output row count does not match inputs")
    if Int(py=embeddings.shape[1]) != expected_cols:
        raise Error("generate_embeddings output columns do not match hidden_size")
    return embeddings


fn _tensor_from_meta(meta_obj: PythonObject) raises -> TensorInfo:
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

    return TensorInfo(ptr_int, s0, s1)

fn _get_tensor(metadata_obj: PythonObject, name: String) raises -> TensorInfo:
    return _tensor_from_meta(metadata_obj.get(name))

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
        layers.append(layer_entry)
        layer_idx += 1

    runtime["layers"] = layers
    return runtime

fn _build_nano_runtime(metadata_obj: PythonObject) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    var runtime = Python.dict()
    runtime["embed_tokens"] = metadata_obj.get("model.embed_tokens.weight")
    runtime["norm"] = metadata_obj.get("model.norm.weight")
    runtime["lm_head"] = metadata_obj.get("lm_head.weight")
    runtime["per_layer_embed"] = metadata_obj.get("model.per_layer_embed.weight")
    runtime["per_layer_projection"] = metadata_obj.get("model.per_layer_embed.projection.weight")
    runtime["per_layer_norm"] = metadata_obj.get("model.per_layer_embed.norm.weight")

    var altup_projections = Python.list()
    var altup_unembeds = Python.list()
    for i in range(3):
        altup_projections.append(metadata_obj.get("model.altup.projection." + String(i) + ".weight"))
        altup_unembeds.append(metadata_obj.get("model.altup.unembed." + String(i) + ".weight"))
    runtime["altup_projections"] = altup_projections
    runtime["altup_unembeds"] = altup_unembeds

    var layers = Python.list()
    var layer_idx = 0
    while True:
        var pfx = "model.layers." + String(layer_idx)
        var layernorm = metadata_obj.get(pfx + ".input_layernorm.weight")
        if not builtins.bool(layernorm):
            break

        var layer_entry = Python.list()
        # Base
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
        # AltUp
        layer_entry.append(metadata_obj.get(pfx + ".altup.router.weight"))
        layer_entry.append(metadata_obj.get(pfx + ".altup.router_norm.weight"))
        layer_entry.append(metadata_obj.get(pfx + ".altup.prediction_coefs"))
        layer_entry.append(metadata_obj.get(pfx + ".altup.correction_coefs"))
        layer_entry.append(metadata_obj.get(pfx + ".altup.output_scale"))
        # Laurel
        layer_entry.append(metadata_obj.get(pfx + ".laurel.down_proj.weight"))
        layer_entry.append(metadata_obj.get(pfx + ".laurel.up_proj.weight"))
        layer_entry.append(metadata_obj.get(pfx + ".laurel.norm.weight"))
        # Per-layer mapping
        layer_entry.append(metadata_obj.get(pfx + ".per_layer_map.gate.weight"))
        layer_entry.append(metadata_obj.get(pfx + ".per_layer_map.projection.weight"))
        layer_entry.append(metadata_obj.get(pfx + ".per_layer_map.norm.weight"))
        layers.append(layer_entry)
        layer_idx += 1

    runtime["layers"] = layers
    return runtime

fn _build_model_from_runtime(runtime_obj: PythonObject) raises -> ModelWeights:
    var m = ModelWeights()

    m.embed_tokens = _tensor_from_meta(runtime_obj["embed_tokens"])
    m.norm = _tensor_from_meta(runtime_obj["norm"])
    m.lm_head = _tensor_from_meta(runtime_obj["lm_head"])

    var layers = runtime_obj["layers"]
    var num_layers = Int(py=Python.import_module("builtins").len(layers))
    for i in range(num_layers):
        var entry = layers[i]
        var layer = LayerWeights()
        layer.input_layernorm = _tensor_from_meta(entry[0])
        layer.post_attention_layernorm = _tensor_from_meta(entry[1])
        layer.q_proj = _tensor_from_meta(entry[2])
        layer.k_proj = _tensor_from_meta(entry[3])
        layer.v_proj = _tensor_from_meta(entry[4])
        layer.o_proj = _tensor_from_meta(entry[5])
        layer.gate_proj = _tensor_from_meta(entry[6])
        layer.up_proj = _tensor_from_meta(entry[7])
        layer.down_proj = _tensor_from_meta(entry[8])
        layer.q_norm = _tensor_from_meta(entry[9])
        layer.k_norm = _tensor_from_meta(entry[10])
        layer.pre_feedforward_layernorm = _tensor_from_meta(entry[11])
        layer.post_feedforward_layernorm = _tensor_from_meta(entry[12])
        m.layers.append(layer^)

    return m^

fn _build_nano_model_from_runtime(runtime_obj: PythonObject) raises -> NanoModelWeights:
    var m = NanoModelWeights()

    m.embed_tokens = _tensor_from_meta(runtime_obj["embed_tokens"])
    m.norm = _tensor_from_meta(runtime_obj["norm"])
    m.lm_head = _tensor_from_meta(runtime_obj["lm_head"])
    m.per_layer_embed = _tensor_from_meta(runtime_obj["per_layer_embed"])
    m.per_layer_projection = _tensor_from_meta(runtime_obj["per_layer_projection"])
    m.per_layer_norm = _tensor_from_meta(runtime_obj["per_layer_norm"])

    var altup_projections = runtime_obj["altup_projections"]
    var altup_unembeds = runtime_obj["altup_unembeds"]
    for i in range(Int(py=Python.import_module("builtins").len(altup_projections))):
        m.altup_projections.append(_tensor_from_meta(altup_projections[i]))
        m.altup_unembeds.append(_tensor_from_meta(altup_unembeds[i]))

    var layers = runtime_obj["layers"]
    var num_layers = Int(py=Python.import_module("builtins").len(layers))
    for i in range(num_layers):
        var entry = layers[i]
        var layer = NanoLayerWeights()
        layer.base.input_layernorm = _tensor_from_meta(entry[0])
        layer.base.post_attention_layernorm = _tensor_from_meta(entry[1])
        layer.base.q_proj = _tensor_from_meta(entry[2])
        layer.base.k_proj = _tensor_from_meta(entry[3])
        layer.base.v_proj = _tensor_from_meta(entry[4])
        layer.base.o_proj = _tensor_from_meta(entry[5])
        layer.base.gate_proj = _tensor_from_meta(entry[6])
        layer.base.up_proj = _tensor_from_meta(entry[7])
        layer.base.down_proj = _tensor_from_meta(entry[8])
        layer.base.q_norm = _tensor_from_meta(entry[9])
        layer.base.k_norm = _tensor_from_meta(entry[10])
        layer.base.pre_feedforward_layernorm = _tensor_from_meta(entry[11])
        layer.base.post_feedforward_layernorm = _tensor_from_meta(entry[12])
        layer.altup.router = _tensor_from_meta(entry[13])
        layer.altup.router_norm = _tensor_from_meta(entry[14])
        layer.altup.prediction_coefs = _tensor_from_meta(entry[15])
        layer.altup.correction_coefs = _tensor_from_meta(entry[16])
        layer.altup.output_scale = _tensor_from_meta(entry[17])
        layer.laurel.down_proj = _tensor_from_meta(entry[18])
        layer.laurel.up_proj = _tensor_from_meta(entry[19])
        layer.laurel.norm = _tensor_from_meta(entry[20])
        layer.per_layer_map.gate = _tensor_from_meta(entry[21])
        layer.per_layer_map.projection = _tensor_from_meta(entry[22])
        layer.per_layer_map.norm = _tensor_from_meta(entry[23])
        m.layers.append(layer^)

    return m^

@always_inline
fn _tensor_is_effectively_zero(t: TensorInfo, eps: Float32 = 1e-8) -> Bool:
    if t.ptr == UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=0):
        return True
    var n = t.shape_0 * t.shape_1
    for i in range(n):
        var v = t.ptr.load(i)
        if v > eps or v < -eps:
            return False
    return True

fn _detect_nano_kv_share_start(model_weights: NanoModelWeights) -> Int:
    var num_layers = len(model_weights.layers)
    var seen_non_zero = False
    for i in range(num_layers):
        var k_zero = _tensor_is_effectively_zero(model_weights.layers[i].base.k_proj)
        var v_zero = _tensor_is_effectively_zero(model_weights.layers[i].base.v_proj)
        if not (k_zero and v_zero):
            seen_non_zero = True
            continue
        if seen_non_zero:
            return i
    return num_layers

fn init_model_mojo(
    metadata_obj: PythonObject
) raises -> PythonObject:
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")
    
    var py_dict = Python.dict()
    py_dict["engine"] = "Mojo Pure Inference Engine"
    
    var arch = _detect_architecture(metadata_obj)
    py_dict["arch"] = arch
    var runtime_obj: PythonObject
    
    var num_layers: Int
    var head_dim: Int
    var num_kv_heads: Int
    var hidden_size: Int
    var intermediate_size: Int
    
    var per_layer_dim: Int = 0
    var vocab_size: Int
    
    if arch == "nano":
        runtime_obj = _build_nano_runtime(metadata_obj)
        var model_weights = _build_nano_model_from_runtime(runtime_obj)
        num_layers = len(model_weights.layers)
        if num_layers == 0:
            raise Error("Invalid Nano model weights: no layers found in metadata")
        head_dim = model_weights.layers[0].base.q_norm.shape_0
        if head_dim == 0:
            head_dim = 256
        num_kv_heads = model_weights.layers[0].base.k_proj.shape_0 // head_dim
        hidden_size = model_weights.embed_tokens.shape_1
        intermediate_size = model_weights.layers[0].base.gate_proj.shape_0
        vocab_size = model_weights.lm_head.shape_0
        per_layer_dim = model_weights.layers[0].per_layer_map.gate.shape_0
        if per_layer_dim <= 0:
            raise Error("Invalid Nano model weights: per_layer_map gate dim must be > 0")
        py_dict["kv_share_start"] = _detect_nano_kv_share_start(model_weights)
    else:
        runtime_obj = _build_standard_runtime(metadata_obj)
        var model_weights = _build_model_from_runtime(runtime_obj)
        num_layers = len(model_weights.layers)
        if num_layers == 0:
            raise Error("Invalid standard model weights: no layers found in metadata")
        head_dim = model_weights.layers[0].q_norm.shape_0
        if head_dim == 0:
            head_dim = 256
        num_kv_heads = model_weights.layers[0].k_proj.shape_0 // head_dim
        hidden_size = model_weights.embed_tokens.shape_1
        intermediate_size = model_weights.layers[0].gate_proj.shape_0
        vocab_size = model_weights.lm_head.shape_0
        py_dict["kv_share_start"] = num_layers
    
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
    py_dict["num_layers"] = num_layers
    py_dict["num_kv_heads"] = num_kv_heads
    py_dict["head_dim"] = head_dim
    py_dict["hidden_size"] = hidden_size
    py_dict["intermediate_size"] = intermediate_size
    py_dict["vocab_size"] = vocab_size
    py_dict["per_layer_dim"] = per_layer_dim
    py_dict["runtime"] = runtime_obj
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
    var arch = String(py=llm["arch"])
    
    if pos >= max_seq_len:
        raise Error("Sequence length exceeded")
        
    var token_id = Int(py=token_id_obj)
    
    var runtime_obj = llm["runtime"]
    var hidden_size = Int(py=llm["hidden_size"])
    var vocab_size = Int(py=llm["vocab_size"])
    var head_dim = Int(py=llm["head_dim"])
    var num_heads: Int
    var num_kv_heads = Int(py=llm["num_kv_heads"])
    var intermediate_size = Int(py=llm["intermediate_size"])
    var kv_share_start = Int(py=llm["kv_share_start"])
    
    var k_cache = llm["k_cache"]
    var v_cache = llm["v_cache"]
    var freqs_cos = llm["freqs_cos"]
    var freqs_sin = llm["freqs_sin"]
    
    var kv_cache_k_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(py=k_cache.__array_interface__["data"][0]))
    var kv_cache_v_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(py=v_cache.__array_interface__["data"][0]))
    var freqs_cos_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(py=freqs_cos.__array_interface__["data"][0]))
    var freqs_sin_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(py=freqs_sin.__array_interface__["data"][0]))
    
    var scratch = List[Float32](length=hidden_size * 160, fill=0.0) # generous scratch space
    var scratch_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(scratch.unsafe_ptr()))
    
    var out_logits = np.zeros(vocab_size, dtype=np.float32)
    var out_logits_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(py=out_logits.__array_interface__["data"][0]))
    
    if arch == "nano":
        var model_weights = _build_nano_model_from_runtime(runtime_obj)
        num_heads = model_weights.layers[0].base.q_proj.shape_0 // head_dim
        var per_layer_dim = Int(py=llm["per_layer_dim"])
        
        forward_nano_step(
            out_logits_ptr,
            token_id,
            pos,
            model_weights,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            per_layer_dim,
            vocab_size,
            freqs_cos_ptr,
            freqs_sin_ptr,
            kv_cache_k_ptr,
            kv_cache_v_ptr,
            max_seq_len,
            kv_share_start,
            scratch_ptr
        )
    else:
        var model_weights = _build_model_from_runtime(runtime_obj)
        num_heads = model_weights.layers[0].q_proj.shape_0 // head_dim
        
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

    var batch_size = Int(py=builtins.len(input_array))
    if batch_size == 0:
        raise Error("inputs must contain at least one row")

    var max_seq_len = 0
    for b in range(batch_size):
        var sl = Int(py=builtins.len(input_array[b]))
        if sl > max_seq_len:
            max_seq_len = sl
            
    if max_seq_len == 0:
        raise Error("inputs must contain at least one token")
    
    var runtime_obj = llm["runtime"]
    var arch = String(py=llm["arch"])
    var num_layers = Int(py=llm["num_layers"])
    var hidden_size = Int(py=llm["hidden_size"])
    var head_dim = Int(py=llm["head_dim"])
    var num_heads: Int
    var num_kv_heads = Int(py=llm["num_kv_heads"])
    var intermediate_size = Int(py=llm["intermediate_size"])
    var kv_share_start = Int(py=llm["kv_share_start"])
    
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
    var scratch = List[Float32](length=hidden_size * 180, fill=0.0) # generous scratch space
    var emb_out = List[Float32](length=batch_size * hidden_size, fill=0.0)
    var input_ids = List[Int32](length=max_seq_len, fill=0)

    # Convert lists to pointers
    var freqs_cos_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(freqs_cos.unsafe_ptr()))
    var freqs_sin_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(freqs_sin.unsafe_ptr()))
    var kv_cache_k_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(kv_cache_k.unsafe_ptr()))
    var kv_cache_v_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(kv_cache_v.unsafe_ptr()))
    var scratch_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(scratch.unsafe_ptr()))
    var emb_out_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(emb_out.unsafe_ptr()))
    var input_ids_ptr = UnsafePointer[Int32, MutExternalOrigin](unsafe_from_address=Int(input_ids.unsafe_ptr()))
    
    var standard_model = ModelWeights()
    var nano_model = NanoModelWeights()
    if arch == "nano":
        nano_model = _build_nano_model_from_runtime(runtime_obj)
        num_heads = nano_model.layers[0].base.q_proj.shape_0 // head_dim
    else:
        standard_model = _build_model_from_runtime(runtime_obj)
        num_heads = standard_model.layers[0].q_proj.shape_0 // head_dim

    # Process each sequence in the batch
    for b in range(batch_size):
        var seq_list = input_array[b]
        var seq_len = Int(py=builtins.len(seq_list))
        
        # Extract input_ids for this batch
        for t in range(seq_len):
            var token_py = seq_list[t]
            input_ids[t] = Int32(Int(py=token_py))
            
        var seq_out_ptr = emb_out_ptr + b * hidden_size
        
        if arch == "nano":
            var per_layer_dim = Int(py=llm["per_layer_dim"])
            
            forward_nano_sequence(
                seq_out_ptr,
                input_ids_ptr,
                seq_len,
                nano_model,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                intermediate_size,
                per_layer_dim,
                freqs_cos_ptr,
                freqs_sin_ptr,
                kv_cache_k_ptr,
                kv_cache_v_ptr,
                max_seq_len,
                kv_share_start,
                scratch_ptr
            )
        else:
            forward_sequence(
                seq_out_ptr,
                input_ids_ptr,
                seq_len,
                standard_model,
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
    
    return _ensure_embedding_matrix(result_np, batch_size, hidden_size, np)

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
