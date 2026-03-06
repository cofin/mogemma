import re

with open("src/mo/mogemma/layers.mojo", "r") as f:
    content = f.read()

# I will just write the implementations to a new file and then append it, 
# after removing the stubs.

# Find the stubs section
stub_start = content.find("@always_inline\nfn forward_per_layer_mapping_gpu(")
if stub_start == -1:
    print("Stubs not found!")
    exit(1)

content_without_stubs = content[:stub_start]

# We will manually define the GPU versions to append.
gpu_code = """
@always_inline
fn _rms_norm_nano_weighted_gpu(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],
    weight_ptr: UnsafePointer[Float32, MutExternalOrigin],
    size: Int,
    eps: Float32 = 1e-6,
):
    var sum_sq: Float32 = 0.0
    for i in range(size):
        var v = x_ptr.load(i)
        sum_sq += v * v
    var inv_rms = 1.0 / sqrt(sum_sq / Float32(size) + eps)
    for i in range(size):
        out_ptr.store(i, x_ptr.load(i) * inv_rms * weight_ptr.load(i))

@always_inline
fn _rms_norm_nano_unit_gpu(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],
    size: Int,
    eps: Float32 = 1e-6,
):
    var sum_sq: Float32 = 0.0
    for i in range(size):
        var v = x_ptr.load(i)
        sum_sq += v * v
    var inv_rms = 1.0 / sqrt(sum_sq / Float32(size) + eps)
    for i in range(size):
        out_ptr.store(i, x_ptr.load(i) * inv_rms)

@always_inline
fn forward_attention_nano_gpu(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    x_ptr: UnsafePointer[Float32, MutExternalOrigin],
    weights: NanoLayerWeights,
    pos: Int,
    hidden_size: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    freqs_cos_ptr: UnsafePointer[Float32, MutExternalOrigin],
    freqs_sin_ptr: UnsafePointer[Float32, MutExternalOrigin],
    kv_cache_k_ptr: UnsafePointer[Float32, MutExternalOrigin],
    kv_cache_v_ptr: UnsafePointer[Float32, MutExternalOrigin],
    max_seq_len: Int,
    write_kv: Bool,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin],
):
    var q_size = num_heads * head_dim
    var kv_size = num_kv_heads * head_dim
    var q_ptr = scratch_ptr
    var k_ptr = scratch_ptr + q_size
    var v_ptr = scratch_ptr + q_size + kv_size

    vec_mat_mul_gpu(q_ptr, x_ptr, weights.base.q_proj.ptr, hidden_size, q_size)
    for h in range(num_heads):
        _rms_norm_nano_weighted_gpu(q_ptr + h * head_dim, q_ptr + h * head_dim, weights.base.q_norm.ptr, head_dim, 1e-6)

    for h in range(num_heads):
        rope_rotate_gpu(q_ptr + h * head_dim, freqs_cos_ptr, freqs_sin_ptr, head_dim)

    if write_kv:
        vec_mat_mul_gpu(k_ptr, x_ptr, weights.base.k_proj.ptr, hidden_size, kv_size)
        vec_mat_mul_gpu(v_ptr, x_ptr, weights.base.v_proj.ptr, hidden_size, kv_size)
        for h in range(num_kv_heads):
            _rms_norm_nano_weighted_gpu(k_ptr + h * head_dim, k_ptr + h * head_dim, weights.base.k_norm.ptr, head_dim, 1e-6)
            _rms_norm_nano_unit_gpu(v_ptr + h * head_dim, v_ptr + h * head_dim, head_dim, 1e-6)
            rope_rotate_gpu(k_ptr + h * head_dim, freqs_cos_ptr, freqs_sin_ptr, head_dim)

        var kv_offset = pos * kv_size
        for i in range(kv_size):
            kv_cache_k_ptr.store(kv_offset + i, k_ptr.load(i))
            kv_cache_v_ptr.store(kv_offset + i, v_ptr.load(i))

    var heads_per_kv = num_heads // num_kv_heads
    var attn_out_ptr = scratch_ptr + q_size + kv_size + kv_size

    for h in range(num_heads):
        var kv_h = h // heads_per_kv
        var q_head_ptr = q_ptr + h * head_dim
        var scores_ptr = attn_out_ptr + num_heads * head_dim

        for t in range(pos + 1):
            var k_head_ptr = kv_cache_k_ptr + t * kv_size + kv_h * head_dim
            var score: Float32 = 0.0
            for d in range(head_dim):
                score += q_head_ptr.load(d) * k_head_ptr.load(d)
            scores_ptr.store(t, score)

        softmax_gpu(scores_ptr, pos + 1)

        var out_head_ptr = attn_out_ptr + h * head_dim
        for d in range(head_dim):
            out_head_ptr.store(d, 0.0)

        for t in range(pos + 1):
            var v_head_ptr = kv_cache_v_ptr + t * kv_size + kv_h * head_dim
            var prob = scores_ptr.load(t)
            for d in range(head_dim):
                var acc = out_head_ptr.load(d)
                out_head_ptr.store(d, acc + prob * v_head_ptr.load(d))

    vec_mat_mul_gpu(out_ptr, attn_out_ptr, weights.base.o_proj.ptr, q_size, hidden_size)

@always_inline
fn forward_per_layer_mapping_gpu(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    active_ptr: UnsafePointer[Float32, MutExternalOrigin],
    per_layer_input_ptr: UnsafePointer[Float32, MutExternalOrigin],
    weights: PerLayerMapWeights,
    hidden_size: Int,
    per_layer_dim: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin]
):
    var gate_out_ptr = scratch_ptr
    var proj_out_ptr = scratch_ptr + per_layer_dim

    vec_mat_mul_gpu(gate_out_ptr, active_ptr, weights.gate.ptr, hidden_size, per_layer_dim)
    var sqrt_2: Float32 = 1.4142135623730951
    for i in range(per_layer_dim):
        var x = gate_out_ptr.load(i)
        var gelu_x = 0.5 * x * (1.0 + erf(x / sqrt_2))
        gate_out_ptr.store(i, gelu_x * per_layer_input_ptr.load(i))

    vec_mat_mul_gpu(proj_out_ptr, gate_out_ptr, weights.projection.ptr, per_layer_dim, hidden_size)
    _rms_norm_nano_weighted_gpu(out_ptr, proj_out_ptr, weights.norm.ptr, hidden_size, 1e-6)


@always_inline
fn _compute_router_modalities_gpu(
    out_modalities_ptr: UnsafePointer[Float32, MutExternalOrigin],
    active_ptr: UnsafePointer[Float32, MutExternalOrigin],
    weights: AltUpWeights,
    hidden_size: Int,
    num_modalities: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin]
):
    var router_in_ptr = scratch_ptr
    _rms_norm_nano_weighted_gpu(router_in_ptr, active_ptr, weights.router_norm.ptr, hidden_size, 1e-6)

    var router_input_scale = 1.0 / Float32(hidden_size)
    for i in range(hidden_size):
        router_in_ptr.store(i, router_in_ptr.load(i) * router_input_scale)

    vec_mat_mul_gpu(out_modalities_ptr, router_in_ptr, weights.router.ptr, hidden_size, num_modalities)
    for m in range(num_modalities):
        out_modalities_ptr.store(m, tanh(out_modalities_ptr.load(m)))

@always_inline
fn forward_altup_predict_gpu(
    out_predictions_ptr: UnsafePointer[Float32, MutExternalOrigin],
    streams_ptr: UnsafePointer[Float32, MutExternalOrigin],
    weights: AltUpWeights,
    hidden_size: Int,
    num_modalities: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin]
):
    var modalities_ptr = scratch_ptr
    var coef_ptr = modalities_ptr + num_modalities
    var router_scratch_ptr = coef_ptr + num_modalities * num_modalities

    _compute_router_modalities_gpu(
        modalities_ptr,
        streams_ptr,
        weights,
        hidden_size,
        num_modalities,
        router_scratch_ptr
    )

    for in_m in range(num_modalities):
        for out_m in range(num_modalities):
            var coef: Float32 = 0.0
            for router_m in range(num_modalities):
                var idx = router_m * num_modalities * num_modalities + out_m * num_modalities + in_m
                coef += weights.prediction_coefs.ptr.load(idx) * modalities_ptr.load(router_m)
            coef_ptr.store(in_m * num_modalities + out_m, coef)

    for out_m in range(num_modalities):
        var out_base = out_predictions_ptr + out_m * hidden_size
        for d in range(hidden_size):
            var pred = streams_ptr.load(out_m * hidden_size + d)
            for in_m in range(num_modalities):
                pred += streams_ptr.load(in_m * hidden_size + d) * coef_ptr.load(in_m * num_modalities + out_m)
            out_base.store(d, pred)

@always_inline
fn forward_altup_correct_gpu(
    out_corrected_ptr: UnsafePointer[Float32, MutExternalOrigin],
    predictions_ptr: UnsafePointer[Float32, MutExternalOrigin],
    activated_ptr: UnsafePointer[Float32, MutExternalOrigin],
    weights: AltUpWeights,
    hidden_size: Int,
    num_modalities: Int,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin]
):
    var modalities_ptr = scratch_ptr
    var corr_ptr = modalities_ptr + num_modalities
    var router_scratch_ptr = corr_ptr + num_modalities

    _compute_router_modalities_gpu(
        modalities_ptr,
        activated_ptr,
        weights,
        hidden_size,
        num_modalities,
        router_scratch_ptr
    )

    for out_m in range(num_modalities):
        var coef: Float32 = 1.0
        for router_m in range(num_modalities):
            coef += weights.correction_coefs.ptr.load(router_m * num_modalities + out_m) * modalities_ptr.load(router_m)
        corr_ptr.store(out_m, coef)

    for d in range(hidden_size):
        var innovation = activated_ptr.load(d) - predictions_ptr.load(d)
        for out_m in range(num_modalities):
            var base_idx = out_m * hidden_size + d
            out_corrected_ptr.store(base_idx, predictions_ptr.load(base_idx) + innovation * corr_ptr.load(out_m))

@always_inline
fn forward_nano_layer_gpu(
    out_streams_ptr: UnsafePointer[Float32, MutExternalOrigin],
    in_streams_ptr: UnsafePointer[Float32, MutExternalOrigin],
    weights: NanoLayerWeights,
    layer_idx: Int,
    per_layer_input_ptr: UnsafePointer[Float32, MutExternalOrigin],
    pos: Int,
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
    num_modalities: Int,
    write_kv: Bool,
    scratch_ptr: UnsafePointer[Float32, MutExternalOrigin]
):
    var predictions_ptr = scratch_ptr                                  # 0..4h
    var corrected_ptr = predictions_ptr + num_modalities * hidden_size # 4h..8h
    var active_ptr = corrected_ptr + num_modalities * hidden_size      # 8h
    var active_norm_ptr = active_ptr + hidden_size                     # 9h
    var laurel_ptr = active_norm_ptr + hidden_size                     # 10h
    var attn_ptr = laurel_ptr + hidden_size                            # 11h
    var attn_norm_ptr = attn_ptr + hidden_size                         # 12h
    var attn_laurel_ptr = attn_norm_ptr + hidden_size                  # 13h
    var ffw_norm_in_ptr = attn_laurel_ptr + hidden_size                # 14h
    var ffw_ptr = ffw_norm_in_ptr + hidden_size                        # 15h
    var ffw_norm_ptr = ffw_ptr + hidden_size                           # 16h
    var activated_ptr = ffw_norm_ptr + hidden_size                     # 17h
    var first_prediction_ptr = activated_ptr + hidden_size             # 18h
    var delta_ptr = first_prediction_ptr + hidden_size                 # 19h
    var altup_scratch_ptr = delta_ptr + hidden_size                    # 20h
    var attn_scratch_ptr = scratch_ptr + hidden_size * 24
    var laurel_scratch_ptr = scratch_ptr + hidden_size * 40
    var ffw_scratch_ptr = scratch_ptr + hidden_size * 44
    var plm_scratch_ptr = scratch_ptr + hidden_size * 48

    forward_altup_predict_gpu(
        predictions_ptr,
        in_streams_ptr,
        weights.altup,
        hidden_size,
        num_modalities,
        altup_scratch_ptr
    )

    for i in range(hidden_size):
        active_ptr.store(i, predictions_ptr.load(i))

    _rms_norm_nano_weighted_gpu(active_norm_ptr, active_ptr, weights.base.input_layernorm.ptr, hidden_size, 1e-6)
    forward_laurel_gpu(
        laurel_ptr,
        active_norm_ptr,
        weights.laurel,
        hidden_size,
        weights.laurel.down_proj.shape_0,
        laurel_scratch_ptr
    )

    forward_attention_nano_gpu(
        attn_ptr, active_norm_ptr, weights, pos, hidden_size, num_heads, num_kv_heads,
        head_dim, freqs_cos_ptr, freqs_sin_ptr, kv_cache_k_ptr, kv_cache_v_ptr, max_seq_len, write_kv, attn_scratch_ptr
    )

    _rms_norm_nano_weighted_gpu(attn_norm_ptr, attn_ptr, weights.base.post_attention_layernorm.ptr, hidden_size, 1e-6)
    var inv_sqrt2: Float32 = 0.7071067811865475
    for i in range(hidden_size):
        attn_laurel_ptr.store(i, (laurel_ptr.load(i) + attn_norm_ptr.load(i)) * inv_sqrt2)

    _rms_norm_nano_weighted_gpu(ffw_norm_in_ptr, attn_laurel_ptr, weights.base.pre_feedforward_layernorm.ptr, hidden_size, 1e-6)
    forward_mlp_gpu(
        ffw_ptr, ffw_norm_in_ptr, weights.base, hidden_size, intermediate_size, ffw_scratch_ptr
    )

    _rms_norm_nano_weighted_gpu(ffw_norm_ptr, ffw_ptr, weights.base.post_feedforward_layernorm.ptr, hidden_size, 1e-6)

    for i in range(hidden_size):
        activated_ptr.store(i, (attn_laurel_ptr.load(i) + ffw_norm_ptr.load(i)) * inv_sqrt2)

    forward_altup_correct_gpu(
        corrected_ptr,
        predictions_ptr,
        activated_ptr,
        weights.altup,
        hidden_size,
        num_modalities,
        altup_scratch_ptr
    )

    for d in range(hidden_size):
        first_prediction_ptr.store(d, predictions_ptr.load(d))

    forward_per_layer_mapping_gpu(
        delta_ptr,
        activated_ptr,
        per_layer_input_ptr,
        weights.per_layer_map,
        hidden_size,
        per_layer_dim,
        plm_scratch_ptr
    )

    for out_m in range(num_modalities):
        var scale = weights.altup.output_scale.ptr.load(out_m)
        for d in range(hidden_size):
            var idx = out_m * hidden_size + d
            var val = corrected_ptr.load(idx)
            if out_m != 0:
                val += delta_ptr.load(d)
            out_streams_ptr.store(idx, val * scale + in_streams_ptr.load(idx))
"""

with open("src/mo/mogemma/layers.mojo", "w") as f:
    f.write(content_without_stubs + gpu_code)
