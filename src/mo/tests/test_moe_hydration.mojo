from std.python import Python, PythonObject

from mogemma.core import _build_moe_from_runtime


def wrap_tensor(arr: PythonObject) raises -> PythonObject:
    return Python.tuple(Int(py=arr.__array_interface__["data"][0]), arr.shape, "float32")


def test_build_moe_from_runtime_populates_all_fields() raises:
    var np = Python.import_module("numpy")
    var metadata = Python.dict()

    var embed = np.zeros(Python.tuple(4, 2), dtype=np.float32)
    var norm = np.zeros(Python.tuple(2), dtype=np.float32)
    var lm_head = np.zeros(Python.tuple(4, 2), dtype=np.float32)
    metadata["model.embed_tokens.weight"] = wrap_tensor(embed)
    metadata["model.norm.weight"] = wrap_tensor(norm)
    metadata["lm_head.weight"] = wrap_tensor(lm_head)

    var pfx = "model.layers.0"
    var input_ln = np.zeros(Python.tuple(2), dtype=np.float32)
    var post_attn_ln = np.zeros(Python.tuple(2), dtype=np.float32)
    var q_proj = np.zeros(Python.tuple(2, 2), dtype=np.float32)
    var k_proj = np.zeros(Python.tuple(2, 2), dtype=np.float32)
    var v_proj = np.zeros(Python.tuple(2, 2), dtype=np.float32)
    var o_proj = np.zeros(Python.tuple(2, 2), dtype=np.float32)
    var q_norm = np.zeros(Python.tuple(2), dtype=np.float32)
    var k_norm = np.zeros(Python.tuple(2), dtype=np.float32)
    var pre_ff = np.zeros(Python.tuple(2), dtype=np.float32)
    var dense_gate = np.zeros(Python.tuple(1, 2), dtype=np.float32)
    var dense_up = np.zeros(Python.tuple(1, 2), dtype=np.float32)
    var dense_down = np.zeros(Python.tuple(2, 1), dtype=np.float32)
    var post_ff1 = np.zeros(Python.tuple(2), dtype=np.float32)
    var pre_ff2 = np.zeros(Python.tuple(2), dtype=np.float32)
    var router_proj = np.zeros(Python.tuple(2, 2), dtype=np.float32)
    var router_scale = np.zeros(Python.tuple(2), dtype=np.float32)
    var per_expert_scale = np.zeros(Python.tuple(2), dtype=np.float32)
    var expert_gate_up = np.zeros(Python.tuple(2, 2, 2), dtype=np.float32)
    var expert_down = np.zeros(Python.tuple(2, 2, 1), dtype=np.float32)
    var post_ff2 = np.zeros(Python.tuple(2), dtype=np.float32)
    var post_ff = np.zeros(Python.tuple(2), dtype=np.float32)
    var skip_scale = np.zeros(Python.tuple(1), dtype=np.float32)

    metadata[pfx + ".input_layernorm.weight"] = wrap_tensor(input_ln)
    metadata[pfx + ".post_attention_layernorm.weight"] = wrap_tensor(post_attn_ln)
    metadata[pfx + ".self_attn.q_proj.weight"] = wrap_tensor(q_proj)
    metadata[pfx + ".self_attn.k_proj.weight"] = wrap_tensor(k_proj)
    metadata[pfx + ".self_attn.v_proj.weight"] = wrap_tensor(v_proj)
    metadata[pfx + ".self_attn.o_proj.weight"] = wrap_tensor(o_proj)
    metadata[pfx + ".self_attn.q_norm.weight"] = wrap_tensor(q_norm)
    metadata[pfx + ".self_attn.k_norm.weight"] = wrap_tensor(k_norm)
    metadata[pfx + ".pre_feedforward_layernorm.weight"] = wrap_tensor(pre_ff)
    metadata[pfx + ".mlp.gate_proj.weight"] = wrap_tensor(dense_gate)
    metadata[pfx + ".mlp.up_proj.weight"] = wrap_tensor(dense_up)
    metadata[pfx + ".mlp.down_proj.weight"] = wrap_tensor(dense_down)
    metadata[pfx + ".post_feedforward_layernorm_1.weight"] = wrap_tensor(post_ff1)
    metadata[pfx + ".pre_feedforward_layernorm_2.weight"] = wrap_tensor(pre_ff2)
    metadata[pfx + ".moe_router.proj.weight"] = wrap_tensor(router_proj)
    metadata[pfx + ".moe_router.scale"] = wrap_tensor(router_scale)
    metadata[pfx + ".moe_router.per_expert_scale"] = wrap_tensor(per_expert_scale)
    metadata[pfx + ".moe_experts.gate_up_proj"] = wrap_tensor(expert_gate_up)
    metadata[pfx + ".moe_experts.down_proj"] = wrap_tensor(expert_down)
    metadata[pfx + ".post_feedforward_layernorm_2.weight"] = wrap_tensor(post_ff2)
    metadata[pfx + ".post_feedforward_layernorm.weight"] = wrap_tensor(post_ff)
    metadata[pfx + ".moe_skip_scale.weight"] = wrap_tensor(skip_scale)

    var model = _build_moe_from_runtime(metadata, 1)
    if len(model.layers) != 1:
        raise Error("Expected exactly one MoE layer")

    var layer = model.layers[0]
    if Int(model.embed_tokens.ptr) != Int(py=embed.__array_interface__["data"][0]):
        raise Error("embed_tokens pointer mismatch")
    if Int(layer.router_proj.ptr) != Int(py=router_proj.__array_interface__["data"][0]):
        raise Error("router_proj pointer mismatch")
    if layer.expert_gate_up_proj.shape_0 != 2 or layer.expert_gate_up_proj.shape_1 != 4:
        raise Error("expert_gate_up_proj shape should flatten trailing dimensions")
    if layer.expert_down_proj.shape_0 != 2 or layer.expert_down_proj.shape_1 != 2:
        raise Error("expert_down_proj shape should flatten trailing dimensions")
    if Int(layer.post_feedforward_layernorm.ptr) != Int(py=post_ff.__array_interface__["data"][0]):
        raise Error("post_feedforward_layernorm pointer mismatch")
    if Int(layer.moe_skip_scale.ptr) != Int(py=skip_scale.__array_interface__["data"][0]):
        raise Error("moe_skip_scale pointer mismatch")


def test_build_moe_from_runtime_keeps_missing_optionals_zero() raises:
    var np = Python.import_module("numpy")
    var metadata = Python.dict()

    metadata["model.embed_tokens.weight"] = wrap_tensor(np.zeros(Python.tuple(4, 2), dtype=np.float32))
    metadata["model.norm.weight"] = wrap_tensor(np.zeros(Python.tuple(2), dtype=np.float32))
    metadata["lm_head.weight"] = wrap_tensor(np.zeros(Python.tuple(4, 2), dtype=np.float32))

    var pfx = "model.layers.0"
    metadata[pfx + ".input_layernorm.weight"] = wrap_tensor(np.zeros(Python.tuple(2), dtype=np.float32))
    metadata[pfx + ".post_attention_layernorm.weight"] = wrap_tensor(np.zeros(Python.tuple(2), dtype=np.float32))
    metadata[pfx + ".self_attn.q_proj.weight"] = wrap_tensor(np.zeros(Python.tuple(2, 2), dtype=np.float32))
    metadata[pfx + ".self_attn.k_proj.weight"] = wrap_tensor(np.zeros(Python.tuple(2, 2), dtype=np.float32))
    metadata[pfx + ".self_attn.v_proj.weight"] = wrap_tensor(np.zeros(Python.tuple(2, 2), dtype=np.float32))
    metadata[pfx + ".self_attn.o_proj.weight"] = wrap_tensor(np.zeros(Python.tuple(2, 2), dtype=np.float32))
    metadata[pfx + ".self_attn.q_norm.weight"] = wrap_tensor(np.zeros(Python.tuple(2), dtype=np.float32))
    metadata[pfx + ".self_attn.k_norm.weight"] = wrap_tensor(np.zeros(Python.tuple(2), dtype=np.float32))
    metadata[pfx + ".pre_feedforward_layernorm.weight"] = wrap_tensor(np.zeros(Python.tuple(2), dtype=np.float32))
    metadata[pfx + ".mlp.gate_proj.weight"] = wrap_tensor(np.zeros(Python.tuple(1, 2), dtype=np.float32))
    metadata[pfx + ".mlp.up_proj.weight"] = wrap_tensor(np.zeros(Python.tuple(1, 2), dtype=np.float32))
    metadata[pfx + ".mlp.down_proj.weight"] = wrap_tensor(np.zeros(Python.tuple(2, 1), dtype=np.float32))
    metadata[pfx + ".post_feedforward_layernorm_1.weight"] = wrap_tensor(np.zeros(Python.tuple(2), dtype=np.float32))
    metadata[pfx + ".pre_feedforward_layernorm_2.weight"] = wrap_tensor(np.zeros(Python.tuple(2), dtype=np.float32))
    metadata[pfx + ".moe_router.proj.weight"] = wrap_tensor(np.zeros(Python.tuple(2, 2), dtype=np.float32))
    metadata[pfx + ".moe_router.scale"] = wrap_tensor(np.zeros(Python.tuple(2), dtype=np.float32))
    metadata[pfx + ".moe_router.per_expert_scale"] = wrap_tensor(np.zeros(Python.tuple(2), dtype=np.float32))
    metadata[pfx + ".moe_experts.gate_up_proj"] = wrap_tensor(np.zeros(Python.tuple(2, 2, 2), dtype=np.float32))
    metadata[pfx + ".moe_experts.down_proj"] = wrap_tensor(np.zeros(Python.tuple(2, 2, 1), dtype=np.float32))
    metadata[pfx + ".post_feedforward_layernorm_2.weight"] = wrap_tensor(np.zeros(Python.tuple(2), dtype=np.float32))

    var model = _build_moe_from_runtime(metadata, 1)
    var layer = model.layers[0]
    if Int(layer.post_feedforward_layernorm.ptr) != 0:
        raise Error("missing post_feedforward_layernorm should stay zero")
    if Int(layer.moe_skip_scale.ptr) != 0:
        raise Error("missing moe_skip_scale should stay zero")


def main() raises:
    test_build_moe_from_runtime_populates_all_fields()
    test_build_moe_from_runtime_keeps_missing_optionals_zero()
    print("test_moe_hydration.mojo passed!")
