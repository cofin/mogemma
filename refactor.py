import os

with open("src/mo/mogemma/core.mojo", "r") as f:
    content = f.read()

# Add alloc import
content = content.replace(
    'from memory import UnsafePointer\n',
    'from memory import UnsafePointer, alloc\n'
)

# 1. Update step_mojo variable extraction
content = content.replace(
    '    var runtime_obj = llm["runtime"]\n',
    '    var runtime_obj = llm["runtime"]\n'
    '    var ptr_std = UnsafePointer[ModelWeights, MutExternalOrigin](unsafe_from_address=Int(py=llm["_descriptor_ptr"]))
'
    '    var ptr_nano = UnsafePointer[NanoModelWeights, MutExternalOrigin](unsafe_from_address=Int(py=llm["_descriptor_ptr"]))
'
)

content = content.replace(
    """_forward_step_nano_gpu_runtime(
                out_logits_ptr,
                token_id,
                pos,
                runtime_obj,"
    ,
    """_forward_step_nano_gpu_runtime(
                out_logits_ptr,
                token_id,
                pos,
                ptr_nano[],"
)
content = content.replace(
    """_forward_step_nano_runtime(
                out_logits_ptr,
                token_id,
                pos,
                runtime_obj,"
    ,
    """_forward_step_nano_runtime(
                out_logits_ptr,
                token_id,
                pos,
                ptr_nano[],"
)
content = content.replace(
    """_forward_step_runtime(
            out_logits_ptr,
            token_id,
            pos,
            runtime_obj,"
    ,
    """_forward_step_runtime(
            out_logits_ptr,
            token_id,
            pos,
            ptr_std[],"
)

# generate_embeddings_mojo extraction
content = content.replace(
    """_generate_embeddings_runtime(
            emb_out_ptr,
            input_ids_ptr,
            batch_size,
            max_seq_len,
            runtime_obj,"
    ,
    """_generate_embeddings_runtime(
            emb_out_ptr,
            input_ids_ptr,
            batch_size,
            max_seq_len,
            ptr_std[],"
)

# 2. Update signatures
content = content.replace(
    """fn _forward_nano_token_hidden_runtime(
    out_hidden_ptr: UnsafePointer[Float32, MutExternalOrigin],
    token_id: Int,
    pos: Int,
    runtime_obj: PythonObject,"
    ,
    """fn _forward_nano_token_hidden_runtime(
    out_hidden_ptr: UnsafePointer[Float32, MutExternalOrigin],
    token_id: Int,
    pos: Int,
    model: NanoModelWeights,"
)
content = content.replace(
    """fn _forward_step_nano_runtime(
    out_logits_ptr: UnsafePointer[Float32, MutExternalOrigin],
    token_id: Int,
    pos: Int,
    runtime_obj: PythonObject,"
    ,
    """fn _forward_step_nano_runtime(
    out_logits_ptr: UnsafePointer[Float32, MutExternalOrigin],
    token_id: Int,
    pos: Int,
    model: NanoModelWeights,"
)
content = content.replace(
    """fn _forward_nano_token_hidden_gpu_runtime(
    out_hidden_ptr: UnsafePointer[Float32, MutExternalOrigin],
    token_id: Int,
    pos: Int,
    runtime_obj: PythonObject,"
    ,
    """fn _forward_nano_token_hidden_gpu_runtime(
    out_hidden_ptr: UnsafePointer[Float32, MutExternalOrigin],
    token_id: Int,
    pos: Int,
    model: NanoModelWeights,"
)
content = content.replace(
    """fn _forward_step_nano_gpu_runtime(
    out_logits_ptr: UnsafePointer[Float32, MutExternalOrigin],
    token_id: Int,
    pos: Int,
    runtime_obj: PythonObject,"
    ,
    """fn _forward_step_nano_gpu_runtime(
    out_logits_ptr: UnsafePointer[Float32, MutExternalOrigin],
    token_id: Int,
    pos: Int,
    model: NanoModelWeights,"
)
content = content.replace(
    """fn _forward_step_runtime(
    out_logits_ptr: UnsafePointer[Float32, MutExternalOrigin],
    token_id: Int,
    pos: Int,
    runtime_obj: PythonObject,"
    ,
    """fn _forward_step_runtime(
    out_logits_ptr: UnsafePointer[Float32, MutExternalOrigin],
    token_id: Int,
    pos: Int,
    model: ModelWeights,"
)
content = content.replace(
    """fn _generate_embeddings_runtime(
    out_emb_ptr: UnsafePointer[Float32, MutExternalOrigin],
    input_ids_ptr: UnsafePointer[Int32, MutExternalOrigin],
    batch_size: Int,
    seq_len: Int,
    runtime_obj: PythonObject,"
    ,
    """fn _generate_embeddings_runtime(
    out_emb_ptr: UnsafePointer[Float32, MutExternalOrigin],
    input_ids_ptr: UnsafePointer[Int32, MutExternalOrigin],
    batch_size: Int,
    seq_len: Int,
    model: ModelWeights,"
)

# 3. Update inner implementations
unpack_std_old = """
    var embed_tokens = _tensor_from_meta(runtime_obj["embed_tokens"])
    var norm = _tensor_from_meta(runtime_obj["norm"])
    var lm_head = _tensor_from_meta(runtime_obj["lm_head"])
    var layers = runtime_obj["layers"]"""

unpack_std_new = """
    var embed_tokens = model.embed_tokens
    var norm = model.norm
    var lm_head = model.lm_head
    var layers = model.layers"""
content = content.replace(unpack_std_old, unpack_std_new)

unpack_nano_old = """
    var builtins = Python.import_module("builtins")
    var runtime_layers = runtime_obj["layers"]
    var first_layer = runtime_layers[0]
    var num_modalities = _tensor_from_meta(first_layer[13]).shape_0

    var current_streams_ptr = scratch_ptr
    var next_streams_ptr = current_streams_ptr + num_modalities * hidden_size
    var per_layer_inputs_ptr = next_streams_ptr + num_modalities * hidden_size
    var layer_scratch_ptr = per_layer_inputs_ptr + Int(py=builtins.len(runtime_layers)) * per_layer_dim

    var num_layers = Int(py=builtins.len(runtime_layers))
    if num_layers == 0:
        raise Error("Invalid Nano runtime: no layers found")

    var embed_tokens = _tensor_from_meta(runtime_obj["embed_tokens"])
    var norm = _tensor_from_meta(runtime_obj["norm"])
    var lm_head = _tensor_from_meta(runtime_obj["lm_head"])
    var per_layer_embed = _tensor_from_meta(runtime_obj["per_layer_embed"])
    var per_layer_projection = _tensor_from_meta(runtime_obj["per_layer_projection"])
    var per_layer_norm = _tensor_from_meta(runtime_obj["per_layer_norm"])
    var per_layer_table_layers = per_layer_embed.shape_1
    var altup_projections = _tensor_list_from_meta(runtime_obj["altup_projections"])
    var altup_unembeds = _tensor_list_from_meta(runtime_obj["altup_unembeds"])"""

unpack_nano_new = """
    var first_layer = model.layers[0]
    var num_modalities = first_layer.altup.router.shape_0

    var current_streams_ptr = scratch_ptr
    var next_streams_ptr = current_streams_ptr + num_modalities * hidden_size
    var per_layer_inputs_ptr = next_streams_ptr + num_modalities * hidden_size
    var layer_scratch_ptr = per_layer_inputs_ptr + len(model.layers) * per_layer_dim

    var num_layers = len(model.layers)
    if num_layers == 0:
        raise Error("Invalid Nano runtime: no layers found")

    var embed_tokens = model.embed_tokens
    var norm = model.norm
    var lm_head = model.lm_head
    var per_layer_embed = model.per_layer_embed
    var per_layer_projection = model.per_layer_projection
    var per_layer_norm = model.per_layer_norm
    var per_layer_table_layers = per_layer_embed.shape_1
    var altup_projections = model.altup_projections
    var altup_unembeds = model.altup_unembeds"""

content = content.replace(unpack_nano_old, unpack_nano_new)

# _generate_embeddings_nano_runtime also has an unpack_nano_old variation
unpack_nano_emb_old = """
    var runtime_layers = runtime_obj["layers"]
    var num_layers = Int(py=builtins.len(runtime_layers))
    if num_layers == 0:
        raise Error("Invalid Nano runtime: no layers found")

    var embed_tokens = _tensor_from_meta(runtime_obj["embed_tokens"])
    var norm = _tensor_from_meta(runtime_obj["norm"])
    var per_layer_embed = _tensor_from_meta(runtime_obj["per_layer_embed"])
    var per_layer_projection = _tensor_from_meta(runtime_obj["per_layer_projection"])
    var per_layer_norm = _tensor_from_meta(runtime_obj["per_layer_norm"])
    var per_layer_table_layers = per_layer_embed.shape_1
    var altup_projections = _tensor_list_from_meta(runtime_obj["altup_projections"])
    var altup_unembeds = _tensor_list_from_meta(runtime_obj["altup_unembeds"])"""

unpack_nano_emb_new = """
    var num_layers = len(model.layers)
    if num_layers == 0:
        raise Error("Invalid Nano runtime: no layers found")

    var embed_tokens = model.embed_tokens
    var norm = model.norm
    var per_layer_embed = model.per_layer_embed
    var per_layer_projection = model.per_layer_projection
    var per_layer_norm = model.per_layer_norm
    var per_layer_table_layers = per_layer_embed.shape_1
    var altup_projections = model.altup_projections
    var altup_unembeds = model.altup_unembeds"""
content = content.replace(unpack_nano_emb_old, unpack_nano_emb_new)

content = content.replace(
    '        var layer_weights = _build_nano_layer_from_runtime_entry(runtime_layers[l])',
    '        var layer_weights = model.layers[l]'
)

content = content.replace(
    '        var layer_weights = _build_standard_layer_from_runtime_entry(layers[l])',
    '        var layer_weights = model.layers[l]'
)

# 4. _init_model_impl_mojo 
init_old = """
    if arch == "nano":
        runtime_obj = _build_nano_runtime(metadata_obj)
        var model_weights = _build_nano_model_from_runtime(runtime_obj)
        nano_model_build_count = 1
        num_layers = len(model_weights.layers)
        if num_layers == 0:
            raise Error("Invalid Nano model weights: no layers found in metadata")
        head_dim = model_weights.layers[0].base.q_norm.shape_0
        if head_dim == 0:
            head_dim = 256
        num_heads = model_weights.layers[0].base.q_proj.shape_0 // head_dim
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
        num_heads = model_weights.layers[0].q_proj.shape_0 // head_dim
        num_kv_heads = model_weights.layers[0].k_proj.shape_0 // head_dim
        hidden_size = model_weights.embed_tokens.shape_1
        intermediate_size = model_weights.layers[0].gate_proj.shape_0
        vocab_size = model_weights.lm_head.shape_0"""

init_new = """
    if arch == "nano":
        runtime_obj = _build_nano_runtime(metadata_obj)
        var model_weights = _build_nano_model_from_runtime(runtime_obj)
        nano_model_build_count = 1
        num_layers = len(model_weights.layers)
        if num_layers == 0:
            raise Error("Invalid Nano model weights: no layers found in metadata")
        head_dim = model_weights.layers[0].base.q_norm.shape_0
        if head_dim == 0:
            head_dim = 256
        num_heads = model_weights.layers[0].base.q_proj.shape_0 // head_dim
        num_kv_heads = model_weights.layers[0].base.k_proj.shape_0 // head_dim
        hidden_size = model_weights.embed_tokens.shape_1
        intermediate_size = model_weights.layers[0].base.gate_proj.shape_0
        vocab_size = model_weights.lm_head.shape_0
        per_layer_dim = model_weights.layers[0].per_layer_map.gate.shape_0
        if per_layer_dim <= 0:
            raise Error("Invalid Nano model weights: per_layer_map gate dim must be > 0")
        py_dict["kv_share_start"] = _detect_nano_kv_share_start(model_weights)
        
        var ptr = alloc[NanoModelWeights](1)
        ptr.init_pointee_move(model_weights^)
        py_dict["_descriptor_ptr"] = Int(ptr)
    else:
        runtime_obj = _build_standard_runtime(metadata_obj)
        var model_weights = _build_model_from_runtime(runtime_obj)
        num_layers = len(model_weights.layers)
        if num_layers == 0:
            raise Error("Invalid standard model weights: no layers found in metadata")
        head_dim = model_weights.layers[0].q_norm.shape_0
        if head_dim == 0:
            head_dim = 256
        num_heads = model_weights.layers[0].q_proj.shape_0 // head_dim
        num_kv_heads = model_weights.layers[0].k_proj.shape_0 // head_dim
        hidden_size = model_weights.embed_tokens.shape_1
        intermediate_size = model_weights.layers[0].gate_proj.shape_0
        vocab_size = model_weights.lm_head.shape_0

        var ptr = alloc[ModelWeights](1)
        ptr.init_pointee_move(model_weights^)
        py_dict["_descriptor_ptr"] = Int(ptr)"""

content = content.replace(init_old, init_new)

# Add free_model_mojo
free_mojo = """

fn free_model_mojo(llm: PythonObject) raises:
    var arch = String(py=llm["arch"])
    var ptr_int = Int(py=llm["_descriptor_ptr"])
    if arch == "nano":
        var ptr = UnsafePointer[NanoModelWeights, MutExternalOrigin](unsafe_from_address=ptr_int)
        ptr.destroy_pointee()
        ptr.free()
    else:
        var ptr = UnsafePointer[ModelWeights, MutExternalOrigin](unsafe_from_address=ptr_int)
        ptr.destroy_pointee()
        ptr.free()
"""
content += free_mojo

# Export free_model_mojo
content = content.replace(
    '        b.def_function[step_mojo]("step")\n        return b.finalize()',
    '        b.def_function[step_mojo]("step")\n        b.def_function[free_model_mojo]("free_model")\n        return b.finalize()'
)

with open("src/mo/mogemma/core.mojo", "w") as f:
    f.write(content)
