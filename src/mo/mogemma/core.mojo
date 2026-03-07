from python import Python, PythonObject
from python.bindings import PythonModuleBuilder
from os import abort
from memory import UnsafePointer, alloc
from math import cos, sin, sqrt
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
    forward_layer,
    forward_nano_layer,
    forward_nano_layer_gpu,
    _collapse_altup_streams,
    _prepare_altup_streams,
    _rms_norm_nano_weighted,
)
from mogemma.ops import rms_norm, vec_mat_mul


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


fn _ensure_embedding_matrix(
    embeddings_obj: PythonObject, expected_rows: Int, expected_cols: Int, np: PythonObject
) raises -> PythonObject:
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


@always_inline
fn _kv_cache_len(batch_size: Int, num_layers: Int, max_seq_len: Int, num_kv_heads: Int, head_dim: Int) -> Int:
    return batch_size * num_layers * max_seq_len * num_kv_heads * head_dim


@always_inline
fn _rope_cache_len(max_seq_len: Int, head_dim: Int) -> Int:
    return max_seq_len * head_dim


@always_inline
fn _step_scratch_len(hidden_size: Int) -> Int:
    return hidden_size * 160


@always_inline
fn _embedding_scratch_len(hidden_size: Int) -> Int:
    return hidden_size * 180


fn _allocate_session_f32(np: PythonObject, length: Int) raises -> PythonObject:
    return np.zeros(length, dtype=np.float32)


fn _allocate_transient_f32(length: Int) -> List[Float32]:
    var values = List[Float32](length=length, fill=0.0)
    return values^


fn _allocate_transient_i32(length: Int) -> List[Int32]:
    var values = List[Int32](length=length, fill=0)
    return values^


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


fn _tensor_list_from_meta(meta_list: PythonObject) raises -> List[TensorInfo]:
    var out = List[TensorInfo]()
    var builtins = Python.import_module("builtins")
    for i in range(Int(py=builtins.len(meta_list))):
        out.append(_tensor_from_meta(meta_list[i]))
    return out^


fn _build_nano_layer_from_runtime_entry(entry: PythonObject) raises -> NanoLayerWeights:
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
    return layer^


fn _build_token_per_layer_inputs_runtime(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],  # [num_layers, per_layer_dim]
    base_stream_ptr: UnsafePointer[Float32, MutExternalOrigin],  # [hidden_size]
    token_id: Int,
    per_layer_embed: TensorInfo,
    per_layer_projection: TensorInfo,
    per_layer_norm: TensorInfo,
    per_layer_table_layers: Int,
    num_layers: Int,
    hidden_size: Int,
    per_layer_dim: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin],
):
    var proj_ptr = scratch_ptr
    var proj_norm_ptr = scratch_ptr + per_layer_dim

    var projection_scale = 1.0 / sqrt(Float32(hidden_size))
    var per_layer_input_scale: Float32 = 0.7071067811865475
    var token_embed_scale = sqrt(Float32(per_layer_dim))

    for l in range(num_layers):
        for p in range(per_layer_dim):
            var acc: Float32 = 0.0
            for d in range(hidden_size):
                var w_idx = d * per_layer_table_layers * per_layer_dim + l * per_layer_dim + p
                acc += base_stream_ptr.load(d) * per_layer_projection.ptr.load(w_idx)
            proj_ptr.store(p, acc * projection_scale)

        _rms_norm_nano_weighted(proj_norm_ptr, proj_ptr, per_layer_norm.ptr, per_layer_dim, 1e-6)

        var embed_base = per_layer_embed.ptr + token_id * per_layer_table_layers * per_layer_dim + l * per_layer_dim
        var out_base = out_ptr + l * per_layer_dim
        for p in range(per_layer_dim):
            var token_embed = embed_base.load(p) * token_embed_scale
            out_base.store(p, (proj_norm_ptr.load(p) + token_embed) * per_layer_input_scale)


fn _forward_nano_token_hidden_runtime(
    out_hidden_ptr: UnsafePointer[Float32, MutExternalOrigin],
    token_ids_ptr: UnsafePointer[Int32, MutExternalOrigin],
    pos: Int,
    runtime_layers: PythonObject,
    num_layers: Int,
    embed_tokens: TensorInfo,
    norm: TensorInfo,
    per_layer_embed: TensorInfo,
    per_layer_projection: TensorInfo,
    per_layer_norm: TensorInfo,
    per_layer_table_layers: Int,
    altup_projections: List[TensorInfo],
    altup_unembeds: List[TensorInfo],
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    intermediate_size: Int,
    per_layer_dim: Int,
    freqs_cos_ptr: UnsafePointer[Float32, MutExternalOrigin],
    freqs_sin_ptr: UnsafePointer[Float32, MutExternalOrigin],
    kv_cache_k_ptr: UnsafePointer[Float32, MutExternalOrigin],
    kv_cache_v_ptr: UnsafePointer[Float32, MutExternalOrigin],
    max_seq_len: Int,
    kv_share_start: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin],
    batch_size: Int = 1,
) raises:
    var first_layer = runtime_layers[0]
    var num_modalities = _tensor_from_meta(first_layer[13]).shape_0

    var current_streams_ptr = scratch_ptr
    var next_streams_ptr = current_streams_ptr + batch_size * num_modalities * hidden_size
    var per_layer_inputs_ptr = next_streams_ptr + batch_size * num_modalities * hidden_size
    var layer_scratch_ptr = per_layer_inputs_ptr + batch_size * num_layers * per_layer_dim
    var collapse_scratch_ptr = layer_scratch_ptr + batch_size * hidden_size * 72
    var stream_init_scratch_ptr = layer_scratch_ptr + batch_size * hidden_size * 68

    var emb_scale = sqrt(Float32(hidden_size))
    for b in range(batch_size):
        var token_id = Int(token_ids_ptr.load(b))
        var emb_row_offset = token_id * hidden_size
        var b_current_stream = current_streams_ptr + b * num_modalities * hidden_size
        for i in range(hidden_size):
            b_current_stream.store(i, embed_tokens.ptr.load(emb_row_offset + i) * emb_scale)

    _prepare_altup_streams(
        current_streams_ptr,
        current_streams_ptr,
        altup_projections,
        hidden_size,
        num_modalities,
        stream_init_scratch_ptr,
        batch_size,
    )

    _build_token_per_layer_inputs_runtime(
        per_layer_inputs_ptr,
        current_streams_ptr,
        token_ids_ptr,
        per_layer_embed,
        per_layer_projection,
        per_layer_norm,
        per_layer_table_layers,
        num_layers,
        hidden_size,
        per_layer_dim,
        layer_scratch_ptr,
        batch_size,
        num_modalities,
    )

    var last_full_kv_layer = kv_share_start - 1
    while last_full_kv_layer >= 0 and ((last_full_kv_layer + 1) % 5) != 0:
        last_full_kv_layer -= 1
    var last_sliding_kv_layer = kv_share_start - 1
    while last_sliding_kv_layer >= 0 and ((last_sliding_kv_layer + 1) % 5) == 0:
        last_sliding_kv_layer -= 1

    for l in range(num_layers):
        var kv_layer_idx = l
        var write_kv = True
        if kv_share_start < num_layers and l >= kv_share_start:
            write_kv = False
            if ((l + 1) % 5) == 0 and last_full_kv_layer >= 0:
                kv_layer_idx = last_full_kv_layer
            elif last_sliding_kv_layer >= 0:
                kv_layer_idx = last_sliding_kv_layer
            else:
                kv_layer_idx = kv_share_start - 1

        var layer_kv_k_ptr = kv_cache_k_ptr + kv_layer_idx * batch_size * max_seq_len * num_kv_heads * head_dim
        var layer_kv_v_ptr = kv_cache_v_ptr + kv_layer_idx * batch_size * max_seq_len * num_kv_heads * head_dim
        var token_freqs_cos_ptr = freqs_cos_ptr + pos * head_dim
        var token_freqs_sin_ptr = freqs_sin_ptr + pos * head_dim

        var layer_per_input_ptr = per_layer_inputs_ptr + l * batch_size * per_layer_dim
        var layer_weights = runtime_layers[l]

        forward_nano_layer(
            next_streams_ptr,
            current_streams_ptr,
            layer_weights,
            l,
            layer_per_input_ptr,
            pos,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            per_layer_dim,
            token_freqs_cos_ptr,
            token_freqs_sin_ptr,
            layer_kv_k_ptr,
            layer_kv_v_ptr,
            max_seq_len,
            num_modalities,
            write_kv,
            layer_scratch_ptr,
            batch_size,
        )

        for i in range(batch_size * num_modalities * hidden_size):
            current_streams_ptr.store(i, next_streams_ptr.load(i))

    _collapse_altup_streams(
        out_hidden_ptr,
        current_streams_ptr,
        altup_unembeds,
        hidden_size,
        num_modalities,
        collapse_scratch_ptr,
        batch_size,
    )
    for b in range(batch_size):
        _rms_norm_nano_weighted(
            out_hidden_ptr + b * hidden_size, out_hidden_ptr + b * hidden_size, norm.ptr, hidden_size, 1e-6
        )


fn _forward_step_nano_runtime(
    out_logits_ptr: UnsafePointer[Float32, MutExternalOrigin],
    token_id: Int,
    pos: Int,
    model: NanoModelWeights,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    intermediate_size: Int,
    per_layer_dim: Int,
    vocab_size: Int,
    freqs_cos_ptr: UnsafePointer[Float32, MutExternalOrigin],
    freqs_sin_ptr: UnsafePointer[Float32, MutExternalOrigin],
    kv_cache_k_ptr: UnsafePointer[Float32, MutExternalOrigin],
    kv_cache_v_ptr: UnsafePointer[Float32, MutExternalOrigin],
    max_seq_len: Int,
    kv_share_start: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin],
) raises:
    """Coordinates the forward pass for a single token in the Nano architecture.

    Orchestrates the AltUp state initialization, sequential Nano layer execution, and final projection to vocabulary logits.
    """
    var builtins = Python.import_module("builtins")
    var runtime_layers = runtime_obj["layers"]
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
    var altup_unembeds = _tensor_list_from_meta(runtime_obj["altup_unembeds"])

    var hidden_ptr = scratch_ptr
    var token_scratch_ptr = scratch_ptr + hidden_size
    _forward_nano_token_hidden_runtime(
        hidden_ptr,
        token_id,
        pos,
        runtime_layers,
        num_layers,
        embed_tokens,
        norm,
        per_layer_embed,
        per_layer_projection,
        per_layer_norm,
        per_layer_table_layers,
        altup_projections,
        altup_unembeds,
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
        token_scratch_ptr,
    )
    vec_mat_mul(out_logits_ptr, hidden_ptr, lm_head.ptr, hidden_size, vocab_size)


fn _forward_sequence_nano_runtime(
    out_emb_ptr: UnsafePointer[Float32, MutExternalOrigin],
    input_ids_ptr: UnsafePointer[Int32, MutExternalOrigin],
    seq_len: Int,
    runtime_obj: PythonObject,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    intermediate_size: Int,
    per_layer_dim: Int,
    freqs_cos_ptr: UnsafePointer[Float32, MutExternalOrigin],
    freqs_sin_ptr: UnsafePointer[Float32, MutExternalOrigin],
    kv_cache_k_ptr: UnsafePointer[Float32, MutExternalOrigin],
    kv_cache_v_ptr: UnsafePointer[Float32, MutExternalOrigin],
    max_seq_len: Int,
    kv_share_start: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin],
    batch_size: Int,
) raises:
    """Coordinates the sequential forward pass for a batch of tokens in the Nano architecture to generate embeddings.

    Passes each token sequentially through the Nano layer stack and aggregates the final hidden states using mean pooling to produce the sequence embedding.
    """
    var builtins = Python.import_module("builtins")
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
    var altup_unembeds = model.altup_unembeds

    var emb_acc_ptr = scratch_ptr
    var token_hidden_ptr = scratch_ptr + batch_size * hidden_size
    var token_ids_buffer_ptr = UnsafePointer[Int32].alloc(batch_size)
    var token_scratch_ptr = scratch_ptr + batch_size * hidden_size * 2
    for i in range(batch_size * hidden_size):
        emb_acc_ptr.store(i, 0.0)

    for t in range(seq_len):
        for b in range(batch_size):
            token_ids_buffer_ptr.store(b, input_ids_ptr.load(b * seq_len + t))

        _forward_nano_token_hidden_runtime(
            token_hidden_ptr,
            token_ids_buffer_ptr,
            t,
            runtime_layers,
            num_layers,
            embed_tokens,
            norm,
            per_layer_embed,
            per_layer_projection,
            per_layer_norm,
            per_layer_table_layers,
            altup_projections,
            altup_unembeds,
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
            token_scratch_ptr,
            batch_size,
        )
        for i in range(batch_size * hidden_size):
            emb_acc_ptr.store(i, emb_acc_ptr.load(i) + token_hidden_ptr.load(i))

    token_ids_buffer_ptr.free()

    var scale = 1.0 / Float32(seq_len)
    for i in range(batch_size * hidden_size):
        out_emb_ptr.store(i, emb_acc_ptr.load(i) * scale)


fn _build_standard_layer_from_runtime_entry(entry: PythonObject) raises -> LayerWeights:
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
    return layer^


fn _forward_step_standard_runtime(
    out_logits_ptr: UnsafePointer[Float32, MutExternalOrigin],
    token_id: Int,
    pos: Int,
    runtime_obj: PythonObject,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    intermediate_size: Int,
    vocab_size: Int,
    freqs_cos_ptr: UnsafePointer[Float32, MutExternalOrigin],
    freqs_sin_ptr: UnsafePointer[Float32, MutExternalOrigin],
    kv_cache_k_ptr: UnsafePointer[Float32, MutExternalOrigin],
    kv_cache_v_ptr: UnsafePointer[Float32, MutExternalOrigin],
    max_seq_len: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin],
) raises:
    """Coordinates the forward pass for a single token in the standard model architecture.

    Extracts token embeddings and sequentially applies each transformer layer, updating the KV cache and computing the final logits.
    """
    var builtins = Python.import_module("builtins")
    var layers = runtime_obj["layers"]
    var num_layers = Int(py=builtins.len(layers))
    var embed_tokens = _tensor_from_meta(runtime_obj["embed_tokens"])
    var norm = _tensor_from_meta(runtime_obj["norm"])
    var lm_head = _tensor_from_meta(runtime_obj["lm_head"])

    var current_state = scratch_ptr
    var next_state = scratch_ptr + hidden_size
    var layer_scratch = scratch_ptr + hidden_size * 2

    var emb_scale = sqrt(Float32(hidden_size))
    var emb_row_offset = token_id * hidden_size
    for i in range(hidden_size):
        current_state.store(i, embed_tokens.ptr.load(emb_row_offset + i) * emb_scale)

    for l in range(num_layers):
        var layer_kv_k_ptr = kv_cache_k_ptr + l * max_seq_len * num_kv_heads * head_dim
        var layer_kv_v_ptr = kv_cache_v_ptr + l * max_seq_len * num_kv_heads * head_dim

        var token_freqs_cos_ptr = freqs_cos_ptr + pos * head_dim
        var token_freqs_sin_ptr = freqs_sin_ptr + pos * head_dim
        var layer_weights = layers[l]

        forward_layer(
            next_state,
            current_state,
            layer_weights,
            pos,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            token_freqs_cos_ptr,
            token_freqs_sin_ptr,
            layer_kv_k_ptr,
            layer_kv_v_ptr,
            max_seq_len,
            layer_scratch,
        )

        for i in range(hidden_size):
            current_state.store(i, next_state.load(i))

    rms_norm(next_state, current_state, norm.ptr, hidden_size, 1e-6)
    vec_mat_mul(out_logits_ptr, next_state, lm_head.ptr, hidden_size, vocab_size)


fn _forward_sequence_standard_runtime(
    out_emb_ptr: UnsafePointer[Float32, MutExternalOrigin],
    input_ids_ptr: UnsafePointer[Int32, MutExternalOrigin],  # [batch_size, seq_len]
    seq_len: Int,
    runtime_obj: PythonObject,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    intermediate_size: Int,
    freqs_cos_ptr: UnsafePointer[Float32, MutExternalOrigin],
    freqs_sin_ptr: UnsafePointer[Float32, MutExternalOrigin],
    kv_cache_k_ptr: UnsafePointer[Float32, MutExternalOrigin],  # [batch_size, max_seq_len, num_kv_heads, head_dim]
    kv_cache_v_ptr: UnsafePointer[Float32, MutExternalOrigin],  # [batch_size, max_seq_len, num_kv_heads, head_dim]
    max_seq_len: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin],
    batch_size: Int,
) raises:
    """Coordinates the sequential forward pass for a batch of tokens in the standard architecture to generate embeddings.

    Passes each token through the transformer layers and aggregates the final hidden states using mean pooling to produce the sequence embedding.
    """
    var builtins = Python.import_module("builtins")
    var layers = runtime_obj["layers"]
    var num_layers = Int(py=builtins.len(layers))
    var embed_tokens = _tensor_from_meta(runtime_obj["embed_tokens"])
    var norm = _tensor_from_meta(runtime_obj["norm"])

    var current_state = scratch_ptr
    var next_state = scratch_ptr + batch_size * hidden_size
    var layer_scratch = scratch_ptr + batch_size * hidden_size * 2
    var emb_acc = layer_scratch + batch_size * hidden_size * 10

    for i in range(batch_size * hidden_size):
        emb_acc.store(i, 0.0)

    var emb_scale = sqrt(Float32(hidden_size))
    for t in range(seq_len):
        for b in range(batch_size):
            var token_id = Int(input_ids_ptr.load(b * seq_len + t))
            var emb_row_offset = token_id * hidden_size
            for i in range(hidden_size):
                current_state.store(b * hidden_size + i, embed_tokens.ptr.load(emb_row_offset + i) * emb_scale)

        for l in range(num_layers):
            var layer_kv_k_ptr = kv_cache_k_ptr + l * batch_size * max_seq_len * num_kv_heads * head_dim
            var layer_kv_v_ptr = kv_cache_v_ptr + l * batch_size * max_seq_len * num_kv_heads * head_dim

            var token_freqs_cos_ptr = freqs_cos_ptr + t * head_dim
            var token_freqs_sin_ptr = freqs_sin_ptr + t * head_dim
            var layer_weights = layers[l]

            forward_layer(
                next_state,
                current_state,
                layer_weights,
                t,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                intermediate_size,
                token_freqs_cos_ptr,
                token_freqs_sin_ptr,
                layer_kv_k_ptr,
                layer_kv_v_ptr,
                max_seq_len,
                layer_scratch,
                batch_size,
            )

            for i in range(batch_size * hidden_size):
                current_state.store(i, next_state.load(i))

        for b in range(batch_size):
            rms_norm(next_state + b * hidden_size, current_state + b * hidden_size, norm.ptr, hidden_size, 1e-6)
            for i in range(hidden_size):
                emb_acc.store(
                    b * hidden_size + i, emb_acc.load(b * hidden_size + i) + next_state.load(b * hidden_size + i)
                )

    var scale = 1.0 / Float32(seq_len)
    for i in range(batch_size * hidden_size):
        out_emb_ptr.store(i, emb_acc.load(i) * scale)


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


fn _init_model_impl_mojo(metadata_obj: PythonObject, device_backend: String) raises -> PythonObject:
    """Internal implementation for initializing the model runtime.

    Detects the model architecture, extracts and constructs the necessary tensor pointers from Python metadata, and allocates the required KV cache, RoPE caches, and scratch buffers for inference.
    """
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")

    var py_dict = Python.dict()
    py_dict["engine"] = "Mojo Pure Inference Engine"

    var arch = _detect_architecture(metadata_obj)
    py_dict["arch"] = arch
    var runtime_obj: PythonObject

    var num_layers: Int
    var head_dim: Int
    var num_heads: Int
    var num_kv_heads: Int
    var hidden_size: Int
    var intermediate_size: Int

    var per_layer_dim: Int = 0
    var vocab_size: Int
    var nano_model_build_count = 0

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
        py_dict["_descriptor_ptr"] = Int(ptr)
        py_dict["kv_share_start"] = num_layers

    var max_seq_len = 8192  # default max seq len

    var session_kv_cache_len = _kv_cache_len(1, num_layers, max_seq_len, num_kv_heads, head_dim)
    var k_cache: PythonObject
    var v_cache: PythonObject

    if device_backend == "cuda" or device_backend == "gpu":
        # Opaque dummy handle for GPU residency. We use Numpy internally for Phase 1-3 testing to avoid segfaults.
        var k_np = _allocate_session_f32(np, session_kv_cache_len)
        var v_np = _allocate_session_f32(np, session_kv_cache_len)
        py_dict["_mock_k"] = k_np
        py_dict["_mock_v"] = v_np
        k_cache = k_np.__array_interface__["data"][0]
        v_cache = v_np.__array_interface__["data"][0]
    else:
        k_cache = _allocate_session_f32(np, session_kv_cache_len)
        v_cache = _allocate_session_f32(np, session_kv_cache_len)

    var rope_cache_len = _rope_cache_len(max_seq_len, head_dim)
    var freqs_cos = _allocate_session_f32(np, rope_cache_len)
    var freqs_sin = _allocate_session_f32(np, rope_cache_len)

    var freqs_cos_ptr = UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(py=freqs_cos.__array_interface__["data"][0])
    )
    var freqs_sin_ptr = UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(py=freqs_sin.__array_interface__["data"][0])
    )

    var base: Float32 = 10000.0
    for t in range(max_seq_len):
        for d in range(head_dim // 2):
            var exp = Float32(d * 2) / Float32(head_dim)
            var inv_freq = 1.0 / (base**exp)
            var freq = Float32(t) * inv_freq
            freqs_cos_ptr.store(t * head_dim + d, cos(freq))
            freqs_sin_ptr.store(t * head_dim + d, sin(freq))

    var step_scratch_len = _step_scratch_len(hidden_size)
    var step_scratch_obj = _allocate_session_f32(np, step_scratch_len)

    py_dict["k_cache"] = k_cache
    py_dict["v_cache"] = v_cache
    py_dict["step_scratch"] = step_scratch_obj
    py_dict["freqs_cos"] = freqs_cos
    py_dict["freqs_sin"] = freqs_sin
    py_dict["max_seq_len"] = max_seq_len
    py_dict["num_layers"] = num_layers
    py_dict["num_heads"] = num_heads
    py_dict["num_kv_heads"] = num_kv_heads
    py_dict["head_dim"] = head_dim
    py_dict["hidden_size"] = hidden_size
    py_dict["intermediate_size"] = intermediate_size
    py_dict["vocab_size"] = vocab_size
    py_dict["session_kv_cache_len"] = session_kv_cache_len
    py_dict["step_scratch_len"] = step_scratch_len
    py_dict["embedding_scratch_len"] = _embedding_scratch_len(hidden_size)
    py_dict["per_layer_dim"] = per_layer_dim
    py_dict["runtime"] = runtime_obj
    py_dict["descriptor_build_count"] = 1
    py_dict["nano_model_build_count"] = nano_model_build_count
    py_dict["pos"] = 0
    return py_dict


fn init_model_mojo(metadata_obj: PythonObject) raises -> PythonObject:
    """Initializes the Mojo inference engine by constructing the model runtime from Python metadata.

    Allocates the KV cache, RoPE positional encodings, and scratch memory spaces for the session.
    """
    return _init_model_impl_mojo(metadata_obj, "cpu")


fn _apply_runtime_init_options(
    llm: PythonObject,
    architecture_overrides_obj: PythonObject,
    device_selection_obj: PythonObject,
) raises:
    var builtins = Python.import_module("builtins")

    if Int(py=builtins.len(architecture_overrides_obj)) > 0:
        llm["architecture_overrides"] = architecture_overrides_obj

    llm["device_selection"] = device_selection_obj
    llm["device_backend"] = device_selection_obj.get("backend")
    llm["device_kind"] = device_selection_obj.get("device_kind")
    llm["device_index"] = device_selection_obj.get("device_index")
    llm["device_request"] = device_selection_obj.get("requested")
    llm["device_availability_source"] = device_selection_obj.get("availability_source")
    llm["device_strict"] = device_selection_obj.get("strict")


fn init_model_with_options_mojo(
    metadata_obj: PythonObject,
    architecture_overrides_obj: PythonObject,
    device_selection_obj: PythonObject,
) raises -> PythonObject:
    """Initializes the model runtime with explicit device selection and architecture overrides.

    Builds the appropriate (Nano or Standard) runtime dictionary, allocates required memory, and applies any specified configuration overrides.
    """
    var backend = String(py=device_selection_obj.get("backend", "cpu"))
    var llm = _init_model_impl_mojo(metadata_obj, backend)
    _apply_runtime_init_options(llm, architecture_overrides_obj, device_selection_obj)
    return llm


fn step_mojo(
    llm: PythonObject,
    token_id_obj: PythonObject,
    temp_obj: PythonObject,
    top_k_obj: PythonObject,
    top_p_obj: PythonObject,
) raises -> PythonObject:
    """Performs a single forward pass step for autoregressive text generation.

    Reads the current generation position, selects the CPU or GPU backend path, dispatches the token through the appropriate Standard or Nano layers, updates internal state/KV cache, and returns the next token logits.
    """
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")

    var pos = Int(py=llm["pos"])
    var max_seq_len = Int(py=llm["max_seq_len"])
    var arch = String(py=llm["arch"])

    if pos >= max_seq_len:
        raise Error("Sequence length exceeded")

    var token_id = Int(py=token_id_obj)

    var runtime_obj = llm["runtime"]
    var ptr_std = UnsafePointer[ModelWeights, MutExternalOrigin](
        unsafe_from_address=Int(py=llm.get("_descriptor_ptr", 0))
    )
    var ptr_nano = UnsafePointer[NanoModelWeights, MutExternalOrigin](
        unsafe_from_address=Int(py=llm.get("_descriptor_ptr", 0))
    )
    var hidden_size = Int(py=llm["hidden_size"])
    var vocab_size = Int(py=llm["vocab_size"])
    var head_dim = Int(py=llm["head_dim"])
    var num_heads = Int(py=llm["num_heads"])
    var num_kv_heads = Int(py=llm["num_kv_heads"])
    var intermediate_size = Int(py=llm["intermediate_size"])
    var kv_share_start = Int(py=llm["kv_share_start"])

    var step_backend = String(py=llm.get("step_backend", ""))
    if step_backend == "":
        var device_backend = String(py=llm.get("device_backend", "cpu"))
        if device_backend == "cuda" or device_backend == "gpu":
            step_backend = "gpu"
            llm["fallback_reason"] = "none"
        else:
            step_backend = "cpu"
            llm["fallback_reason"] = "requested"
        llm["step_backend"] = step_backend

    # Latch counters
    var launch_counter = Int(py=llm.get("debug_launch_count", 0))
    llm["debug_launch_count"] = launch_counter + 1

    var k_cache = llm["k_cache"]
    var v_cache = llm["v_cache"]
    var freqs_cos = llm["freqs_cos"]
    var freqs_sin = llm["freqs_sin"]

    var freqs_cos_ptr = UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(py=freqs_cos.__array_interface__["data"][0])
    )
    var freqs_sin_ptr = UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(py=freqs_sin.__array_interface__["data"][0])
    )

    var step_scratch_len = Int(py=llm["step_scratch_len"])
    var scratch_obj = llm["step_scratch"]
    var scratch_ptr: UnsafePointer[Float32, MutExternalOrigin]

    var kv_cache_k_ptr: UnsafePointer[Float32, MutExternalOrigin]
    var kv_cache_v_ptr: UnsafePointer[Float32, MutExternalOrigin]

    if step_backend == "cuda" or step_backend == "gpu":
        kv_cache_k_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(py=k_cache))
        kv_cache_v_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(py=v_cache))
        # scratch_obj is a numpy array for scaffolding
        scratch_ptr = UnsafePointer[Float32, MutExternalOrigin](
            unsafe_from_address=Int(py=scratch_obj.__array_interface__["data"][0])
        )
    else:
        # standard numpy cache
        kv_cache_k_ptr = UnsafePointer[Float32, MutExternalOrigin](
            unsafe_from_address=Int(py=k_cache.__array_interface__["data"][0])
        )
        kv_cache_v_ptr = UnsafePointer[Float32, MutExternalOrigin](
            unsafe_from_address=Int(py=v_cache.__array_interface__["data"][0])
        )

        # for cpu, scratch_obj is a numpy array
        scratch_ptr = UnsafePointer[Float32, MutExternalOrigin](
            unsafe_from_address=Int(py=scratch_obj.__array_interface__["data"][0])
        )

    var out_logits = np.zeros(vocab_size, dtype=np.float32)
    var out_logits_ptr = UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address=Int(py=out_logits.__array_interface__["data"][0])
    )

    if arch == "nano":
        var per_layer_dim = Int(py=llm["per_layer_dim"])
        if step_backend == "cuda" or step_backend == "gpu":
            _forward_step_nano_gpu_runtime(
                out_logits_ptr,
                token_id,
                pos,
                ptr_nano[],
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
                scratch_ptr,
            )
        else:
            _forward_step_nano_runtime(
                out_logits_ptr,
                token_id,
                pos,
                ptr_nano[],
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
                scratch_ptr,
            )
    else:
        # Since we use CPU polyfills for GPU tests during Phase 1-3, we can just call the standard runtime
        # which will eventually dispatch to GPU kernels or CPU kernels based on the backend latch.
        # But for now, we just pass the mock pointers down.
        _forward_step_standard_runtime(
            out_logits_ptr,
            token_id,
            pos,
            runtime_obj,
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
            scratch_ptr,
        )

    llm["pos"] = pos + 1

    return _ensure_step_logits(out_logits, np)


fn generate_embeddings_mojo(
    llm: PythonObject,
    input_array: PythonObject,
) raises -> PythonObject:
    """Generates mean-pooled embeddings for a batch of input token sequences.

    Iterates over the sequences, processing them through the initialized runtime model to produce sequence-level continuous vector representations.
    """
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
    var ptr_std = UnsafePointer[ModelWeights, MutExternalOrigin](
        unsafe_from_address=Int(py=llm.get("_descriptor_ptr", 0))
    )
    var ptr_nano = UnsafePointer[NanoModelWeights, MutExternalOrigin](
        unsafe_from_address=Int(py=llm.get("_descriptor_ptr", 0))
    )
    var arch = String(py=llm["arch"])
    var num_layers = Int(py=llm["num_layers"])
    var hidden_size = Int(py=llm["hidden_size"])
    var head_dim = Int(py=llm["head_dim"])
    var num_heads = Int(py=llm["num_heads"])
    var num_kv_heads = Int(py=llm["num_kv_heads"])
    var intermediate_size = Int(py=llm["intermediate_size"])
    var kv_share_start = Int(py=llm["kv_share_start"])

    # Prefer session-level RoPE buffers computed in init_model.
    var freqs_cos = llm["freqs_cos"]
    var freqs_sin = llm["freqs_sin"]
    var runtime_max_seq_len = Int(py=llm["max_seq_len"])

    var freqs_cos_local = List[Float32]()
    var freqs_sin_local = List[Float32]()
    var freqs_cos_ptr: UnsafePointer[Float32, MutExternalOrigin]
    var freqs_sin_ptr: UnsafePointer[Float32, MutExternalOrigin]
    if max_seq_len <= runtime_max_seq_len:
        freqs_cos_ptr = UnsafePointer[Float32, MutExternalOrigin](
            unsafe_from_address=Int(py=freqs_cos.__array_interface__["data"][0])
        )
        freqs_sin_ptr = UnsafePointer[Float32, MutExternalOrigin](
            unsafe_from_address=Int(py=freqs_sin.__array_interface__["data"][0])
        )
    else:
        freqs_cos_local = List[Float32](length=max_seq_len * head_dim, fill=0.0)
        freqs_sin_local = List[Float32](length=max_seq_len * head_dim, fill=0.0)
        var base: Float32 = 10000.0
        for t in range(max_seq_len):
            for d in range(head_dim // 2):
                var exp = Float32(d * 2) / Float32(head_dim)
                var inv_freq = 1.0 / (base**exp)
                var freq = Float32(t) * inv_freq
                freqs_cos_local[t * head_dim + d] = cos(freq)
                freqs_sin_local[t * head_dim + d] = sin(freq)
        freqs_cos_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(freqs_cos_local.unsafe_ptr()))
        freqs_sin_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(freqs_sin_local.unsafe_ptr()))

    # Allocations for intermediate state
    var embedding_kv_cache_len = _kv_cache_len(batch_size, num_layers, max_seq_len, num_kv_heads, head_dim)
    var kv_cache_k = _allocate_transient_f32(embedding_kv_cache_len)
    var kv_cache_v = _allocate_transient_f32(embedding_kv_cache_len)
    var embedding_scratch_len = Int(py=llm["embedding_scratch_len"])
    var scratch = _allocate_transient_f32(batch_size * embedding_scratch_len)  # generous scratch space
    var emb_out = _allocate_transient_f32(batch_size * hidden_size)
    var input_ids = _allocate_transient_i32(batch_size * max_seq_len)

    # Convert lists to pointers
    var kv_cache_k_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(kv_cache_k.unsafe_ptr()))
    var kv_cache_v_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(kv_cache_v.unsafe_ptr()))
    var scratch_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(scratch.unsafe_ptr()))
    var emb_out_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(emb_out.unsafe_ptr()))
    var input_ids_ptr = UnsafePointer[Int32, MutExternalOrigin](unsafe_from_address=Int(input_ids.unsafe_ptr()))

    var seq_len = Int(py=builtins.len(input_array[0]))

    if arch == "nano":
        var per_layer_dim = Int(py=llm["per_layer_dim"])
        for b in range(batch_size):
            var seq_list = input_array[b]
            for t in range(seq_len):
                var token_py = seq_list[t]
                input_ids[b * seq_len + t] = Int32(Int(py=token_py))

        _forward_sequence_nano_runtime(
            emb_out_ptr,
            input_ids_ptr,
            seq_len,
            runtime_obj,
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
            scratch_ptr,
            batch_size,
        )
    else:
        for b in range(batch_size):
            var seq_list = input_array[b]
            for t in range(seq_len):
                var token_py = seq_list[t]
                input_ids[b * seq_len + t] = Int32(Int(py=token_py))

        _forward_sequence_standard_runtime(
            emb_out_ptr,
            input_ids_ptr,
            seq_len,
            runtime_obj,
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
            scratch_ptr,
            batch_size,
        )

    # Return as numpy array
    var result_np = np.zeros(Python.tuple(batch_size, hidden_size), dtype=np.float32)
    # Copy from emb_out back to numpy
    for b in range(batch_size):
        for i in range(hidden_size):
            var val = emb_out[b * hidden_size + i]
            result_np[b][i] = val

    # We must ensure refs to Mojo lists are kept alive till here.
    if len(freqs_cos_local) > 0:
        _ = freqs_cos_local[0]
    if len(freqs_sin_local) > 0:
        _ = freqs_sin_local[0]
    _ = kv_cache_k[0]
    _ = kv_cache_v[0]
    _ = scratch[0]
    _ = emb_out[0]
    _ = input_ids[0]

    return _ensure_embedding_matrix(result_np, batch_size, hidden_size, np)


fn _forward_nano_token_hidden_gpu_runtime(
    out_hidden_ptr: UnsafePointer[Float32, MutExternalOrigin],
    token_id: Int,
    pos: Int,
    runtime_layers: PythonObject,
    num_layers: Int,
    embed_tokens: TensorInfo,
    norm: TensorInfo,
    per_layer_embed: TensorInfo,
    per_layer_projection: TensorInfo,
    per_layer_norm: TensorInfo,
    per_layer_table_layers: Int,
    altup_projections: List[TensorInfo],
    altup_unembeds: List[TensorInfo],
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    intermediate_size: Int,
    per_layer_dim: Int,
    freqs_cos_ptr: UnsafePointer[Float32, MutExternalOrigin],
    freqs_sin_ptr: UnsafePointer[Float32, MutExternalOrigin],
    kv_cache_k_ptr: UnsafePointer[Float32, MutExternalOrigin],
    kv_cache_v_ptr: UnsafePointer[Float32, MutExternalOrigin],
    max_seq_len: Int,
    kv_share_start: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin],
) raises:
    var first_layer = runtime_layers[0]
    var num_modalities = _tensor_from_meta(first_layer[13]).shape_0

    var current_streams_ptr = scratch_ptr
    var next_streams_ptr = current_streams_ptr + num_modalities * hidden_size
    var per_layer_inputs_ptr = next_streams_ptr + num_modalities * hidden_size
    var layer_scratch_ptr = per_layer_inputs_ptr + num_layers * per_layer_dim
    var collapse_scratch_ptr = layer_scratch_ptr + hidden_size * 72
    var stream_init_scratch_ptr = layer_scratch_ptr + hidden_size * 68

    var emb_scale = sqrt(Float32(hidden_size))
    var emb_row_offset = token_id * hidden_size
    for i in range(hidden_size):
        current_streams_ptr.store(i, embed_tokens.ptr.load(emb_row_offset + i) * emb_scale)

    _prepare_altup_streams(
        current_streams_ptr,
        current_streams_ptr,
        altup_projections,
        hidden_size,
        num_modalities,
        stream_init_scratch_ptr,
    )

    _build_token_per_layer_inputs_runtime(
        per_layer_inputs_ptr,
        current_streams_ptr,
        token_id,
        per_layer_embed,
        per_layer_projection,
        per_layer_norm,
        per_layer_table_layers,
        num_layers,
        hidden_size,
        per_layer_dim,
        layer_scratch_ptr,
    )

    var last_full_kv_layer = kv_share_start - 1
    while last_full_kv_layer >= 0 and ((last_full_kv_layer + 1) % 5) != 0:
        last_full_kv_layer -= 1
    var last_sliding_kv_layer = kv_share_start - 1
    while last_sliding_kv_layer >= 0 and ((last_sliding_kv_layer + 1) % 5) == 0:
        last_sliding_kv_layer -= 1

    for l in range(num_layers):
        var kv_layer_idx = l
        var write_kv = True
        if kv_share_start < num_layers and l >= kv_share_start:
            write_kv = False
            if ((l + 1) % 5) == 0 and last_full_kv_layer >= 0:
                kv_layer_idx = last_full_kv_layer
            elif last_sliding_kv_layer >= 0:
                kv_layer_idx = last_sliding_kv_layer
            else:
                kv_layer_idx = kv_share_start - 1

        var layer_kv_k_ptr = kv_cache_k_ptr + kv_layer_idx * max_seq_len * num_kv_heads * head_dim
        var layer_kv_v_ptr = kv_cache_v_ptr + kv_layer_idx * max_seq_len * num_kv_heads * head_dim
        var token_freqs_cos_ptr = freqs_cos_ptr + pos * head_dim
        var token_freqs_sin_ptr = freqs_sin_ptr + pos * head_dim
        var layer_per_input_ptr = per_layer_inputs_ptr + l * per_layer_dim
        var layer_weights = runtime_layers[l]

        forward_nano_layer_gpu(
            next_streams_ptr,
            current_streams_ptr,
            layer_weights,
            l,
            layer_per_input_ptr,
            pos,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            per_layer_dim,
            token_freqs_cos_ptr,
            token_freqs_sin_ptr,
            layer_kv_k_ptr,
            layer_kv_v_ptr,
            max_seq_len,
            num_modalities,
            write_kv,
            layer_scratch_ptr,
        )

        for i in range(num_modalities * hidden_size):
            current_streams_ptr.store(i, next_streams_ptr.load(i))

    _collapse_altup_streams(
        out_hidden_ptr, current_streams_ptr, altup_unembeds, hidden_size, num_modalities, collapse_scratch_ptr
    )
    # Using CPU norm here for now, polyfill is fine
    _rms_norm_nano_weighted(out_hidden_ptr, out_hidden_ptr, norm.ptr, hidden_size, 1e-6)


fn _forward_step_nano_gpu_runtime(
    out_logits_ptr: UnsafePointer[Float32, MutExternalOrigin],
    token_id: Int,
    pos: Int,
    model: NanoModelWeights,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    intermediate_size: Int,
    per_layer_dim: Int,
    vocab_size: Int,
    freqs_cos_ptr: UnsafePointer[Float32, MutExternalOrigin],
    freqs_sin_ptr: UnsafePointer[Float32, MutExternalOrigin],
    kv_cache_k_ptr: UnsafePointer[Float32, MutExternalOrigin],
    kv_cache_v_ptr: UnsafePointer[Float32, MutExternalOrigin],
    max_seq_len: Int,
    kv_share_start: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin],
) raises:
    var builtins = Python.import_module("builtins")
    var runtime_layers = runtime_obj["layers"]
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
    var altup_unembeds = _tensor_list_from_meta(runtime_obj["altup_unembeds"])

    var hidden_ptr = scratch_ptr
    var token_scratch_ptr = scratch_ptr + hidden_size
    _forward_nano_token_hidden_gpu_runtime(
        hidden_ptr,
        token_id,
        pos,
        runtime_layers,
        num_layers,
        embed_tokens,
        norm,
        per_layer_embed,
        per_layer_projection,
        per_layer_norm,
        per_layer_table_layers,
        altup_projections,
        altup_unembeds,
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
        token_scratch_ptr,
    )
    # CPU polyfill for matmul
    from mogemma.ops_gpu import vec_mat_mul_gpu

    vec_mat_mul_gpu(out_logits_ptr, hidden_ptr, lm_head.ptr, hidden_size, vocab_size)


@export
fn PyInit__core() -> PythonObject:
    try:
        var b = PythonModuleBuilder("_core")
        b.def_function[init_model_mojo]("init_model")
        b.def_function[init_model_with_options_mojo]("init_model_with_options")
        b.def_function[generate_embeddings_mojo]("generate_embeddings")
        b.def_function[step_mojo]("step")
        b.def_function[free_model_mojo]("free_model")
        return b.finalize()
    except e:
        abort(String("failed to create Python module: ", e))


fn free_model_mojo(llm: PythonObject) raises:
    """Frees the unmanaged memory allocated for the Mojo model descriptor.

    Reads the architecture type and raw pointer address from the runtime dictionary, casts it to the correct pointer type, and frees it to prevent memory leaks.
    """
    var arch = String(py=llm["arch"])
    var ptr_int = Int(py=llm.get("_descriptor_ptr", 0))
    if ptr_int == 0:
        return
    if arch == "nano":
        var ptr = UnsafePointer[NanoModelWeights, MutExternalOrigin](unsafe_from_address=ptr_int)
        ptr.destroy_pointee()
        ptr.free()
    else:
        var ptr = UnsafePointer[ModelWeights, MutExternalOrigin](unsafe_from_address=ptr_int)
        ptr.destroy_pointee()
        ptr.free()
