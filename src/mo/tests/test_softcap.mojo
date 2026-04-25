from std.collections import List
from std.memory import UnsafePointer
from std.testing import assert_almost_equal

from mogemma.layers import (
    forward_full_attention,
    forward_gemma4_step,
    forward_moe_layer,
    forward_sliding_attention,
)
from mogemma.model import (
    KVCache,
    LAYER_TYPE_FULL,
    LAYER_TYPE_SLIDING,
    LayerWeights,
    ModelWeights,
    MoELayerWeights,
    RoPETables,
    TensorInfo,
)
from mogemma.ops import CPUBackend, ComputeBackend


def _ptr(ref values: List[Float32]) -> UnsafePointer[Float32, MutExternalOrigin]:
    return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(values.unsafe_ptr()))


def _zeros(size: Int) -> List[Float32]:
    return List[Float32](length=size, fill=0.0)


def _assert_true(value: Bool, message: String) raises:
    if not value:
        raise Error(message)


struct TrackingBackend(ComputeBackend, Copyable, ImplicitlyCopyable, Movable):
    var softcap_calls: Int
    var softmax_calls: Int
    var softcap_before_softmax: Bool
    var softmax_calls_at_first_softcap: Int

    def __init__(out self):
        self.softcap_calls = 0
        self.softmax_calls = 0
        self.softcap_before_softmax = False
        self.softmax_calls_at_first_softcap = -1

    def vec_mat_mul(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_ptr: UnsafePointer[Float32, MutAnyOrigin],
        w_ptr: UnsafePointer[Float32, MutAnyOrigin],
        in_dim: Int,
        out_dim: Int,
    ):
        for i in range(out_dim):
            out_ptr.store(i, 1.0)
        _ = x_ptr
        _ = w_ptr
        _ = in_dim

    def vec_mat_mul_i8(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_ptr: UnsafePointer[Float32, MutAnyOrigin],
        w_ptr: UnsafePointer[Int8, MutAnyOrigin],
        scale_ptr: UnsafePointer[Float32, MutAnyOrigin],
        in_dim: Int,
        out_dim: Int,
    ):
        for i in range(out_dim):
            out_ptr.store(i, 1.0)
        _ = x_ptr
        _ = w_ptr
        _ = scale_ptr
        _ = in_dim

    def mat_mat_mul(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_ptr: UnsafePointer[Float32, MutAnyOrigin],
        w_ptr: UnsafePointer[Float32, MutAnyOrigin],
        batch_size: Int,
        in_dim: Int,
        out_dim: Int,
    ):
        for i in range(batch_size * out_dim):
            out_ptr.store(i, 1.0)
        _ = x_ptr
        _ = w_ptr
        _ = in_dim

    def rms_norm(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_ptr: UnsafePointer[Float32, MutAnyOrigin],
        weight_ptr: UnsafePointer[Float32, MutAnyOrigin],
        size: Int,
        eps: Float32,
    ):
        for i in range(size):
            out_ptr.store(i, x_ptr.load(i))
        _ = weight_ptr
        _ = eps

    def softmax(mut self, vec_ptr: UnsafePointer[Float32, MutAnyOrigin], size: Int):
        if self.softcap_calls > 0:
            self.softcap_before_softmax = True
        self.softmax_calls += 1
        _ = vec_ptr
        _ = size

    def softcap(
        mut self,
        vec_ptr: UnsafePointer[Float32, MutAnyOrigin],
        size: Int,
        cap: Float32,
    ):
        if self.softcap_calls == 0:
            self.softmax_calls_at_first_softcap = self.softmax_calls
        self.softcap_calls += 1
        _ = vec_ptr
        _ = size
        _ = cap

    def rope_rotate(
        mut self,
        vec_ptr: UnsafePointer[Float32, MutAnyOrigin],
        cos_ptr: UnsafePointer[Float32, MutAnyOrigin],
        sin_ptr: UnsafePointer[Float32, MutAnyOrigin],
        head_dim: Int,
    ):
        _ = vec_ptr
        _ = cos_ptr
        _ = sin_ptr
        _ = head_dim

    def geglu(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        gate_ptr: UnsafePointer[Float32, MutAnyOrigin],
        up_ptr: UnsafePointer[Float32, MutAnyOrigin],
        size: Int,
    ):
        _ = out_ptr
        _ = gate_ptr
        _ = up_ptr
        _ = size

    def gelu(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_ptr: UnsafePointer[Float32, MutAnyOrigin],
        size: Int,
    ):
        for i in range(size):
            out_ptr.store(i, x_ptr.load(i))

    def copy(
        mut self,
        dst_ptr: UnsafePointer[Float32, MutAnyOrigin],
        src_ptr: UnsafePointer[Float32, MutAnyOrigin],
        size: Int,
    ):
        for i in range(size):
            dst_ptr.store(i, src_ptr.load(i))

    def embed_lookup(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        embed_table_ptr: UnsafePointer[Float32, MutAnyOrigin],
        token_id: Int,
        hidden_size: Int,
        scale: Float32,
    ):
        var src = embed_table_ptr + token_id * hidden_size
        for i in range(hidden_size):
            out_ptr.store(i, src.load(i) * scale)

    def vector_add(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        a_ptr: UnsafePointer[Float32, MutAnyOrigin],
        b_ptr: UnsafePointer[Float32, MutAnyOrigin],
        size: Int,
    ):
        for i in range(size):
            out_ptr.store(i, a_ptr.load(i) + b_ptr.load(i))

    def vector_add_scaled(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        a_ptr: UnsafePointer[Float32, MutAnyOrigin],
        b_ptr: UnsafePointer[Float32, MutAnyOrigin],
        scale: Float32,
        size: Int,
    ):
        for i in range(size):
            out_ptr.store(i, a_ptr.load(i) + b_ptr.load(i) * scale)

    def average_pool_2d(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_ptr: UnsafePointer[Float32, MutAnyOrigin],
        grid_h: Int,
        grid_w: Int,
        hidden_size: Int,
        kernel: Int,
    ):
        _ = out_ptr
        _ = x_ptr
        _ = grid_h
        _ = grid_w
        _ = hidden_size
        _ = kernel

    def top_k(
        mut self,
        values_ptr: UnsafePointer[Float32, MutAnyOrigin],
        k: Int,
        size: Int,
        out_indices_ptr: UnsafePointer[Int32, MutAnyOrigin],
        out_values_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ):
        for i in range(k):
            out_indices_ptr.store(i, Int32(i))
            out_values_ptr.store(i, values_ptr.load(i))
        _ = size

    def kv_write(
        mut self,
        dst_ptr: UnsafePointer[Float32, MutAnyOrigin],
        src_ptr: UnsafePointer[Float32, MutAnyOrigin],
        kv_size: Int,
        pos: Int,
        cache_size: Int,
        layer_offset: Int,
        is_full: Bool,
    ):
        var dst = dst_ptr + layer_offset
        for i in range(kv_size):
            dst.store(i, src_ptr.load(i))
        _ = pos
        _ = cache_size
        _ = is_full

    def attention_scores(
        mut self,
        scores_ptr: UnsafePointer[Float32, MutAnyOrigin],
        q_ptr: UnsafePointer[Float32, MutAnyOrigin],
        k_cache_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_heads: Int,
        num_kv_heads: Int,
        head_dim: Int,
        valid_len: Int,
        kv_size: Int,
        scale: Float32,
    ):
        for i in range(num_heads * valid_len):
            scores_ptr.store(i, 100.0)
        _ = q_ptr
        _ = k_cache_ptr
        _ = num_kv_heads
        _ = head_dim
        _ = kv_size
        _ = scale

    def attention_value_accum(
        mut self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        scores_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_cache_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_heads: Int,
        num_kv_heads: Int,
        head_dim: Int,
        valid_len: Int,
        kv_size: Int,
    ):
        for i in range(num_heads * head_dim):
            out_ptr.store(i, 1.0)
        _ = scores_ptr
        _ = v_cache_ptr
        _ = num_kv_heads
        _ = valid_len
        _ = kv_size


def _call_sliding_attention(attn_logit_softcapping: Float32) raises -> TrackingBackend:
    var layer_types = List[UInt8](length=1, fill=UInt8(LAYER_TYPE_SLIDING))
    var kv_cache = KVCache(
        1,
        1,
        1,
        4,
        4,
        UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(layer_types.unsafe_ptr())),
    )
    var rope_tables = RoPETables(1, 1.0, 4, 4)
    var x = List[Float32](length=2, fill=1.0)
    var out = _zeros(2)
    var scratch = _zeros(64)
    var q_proj = List[Float32](length=4, fill=1.0)
    var k_proj = List[Float32](length=2, fill=1.0)
    var v_proj = List[Float32](length=2, fill=1.0)
    var o_proj = List[Float32](length=4, fill=1.0)
    var weights = LayerWeights()
    weights.q_proj = TensorInfo(Int(q_proj.unsafe_ptr()), 2, 2)
    weights.k_proj = TensorInfo(Int(k_proj.unsafe_ptr()), 1, 2)
    weights.v_proj = TensorInfo(Int(v_proj.unsafe_ptr()), 1, 2)
    weights.o_proj = TensorInfo(Int(o_proj.unsafe_ptr()), 2, 2)
    var backend = TrackingBackend()

    forward_sliding_attention(
        backend,
        _ptr(out),
        _ptr(x),
        weights,
        0,
        0,
        2,
        2,
        1,
        1,
        kv_cache,
        rope_tables,
        False,
        attn_logit_softcapping,
        _ptr(scratch),
    )

    _ = layer_types
    _ = x[0]
    _ = out[0]
    _ = scratch[0]
    _ = q_proj[0]
    _ = k_proj[0]
    _ = v_proj[0]
    _ = o_proj[0]
    return backend^


def _call_full_attention(attn_logit_softcapping: Float32) raises -> TrackingBackend:
    var layer_types = List[UInt8](length=1, fill=UInt8(LAYER_TYPE_FULL))
    var kv_cache = KVCache(
        1,
        1,
        1,
        4,
        4,
        UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(layer_types.unsafe_ptr())),
    )
    var rope_tables = RoPETables(1, 1.0, 4, 4)
    var x = List[Float32](length=2, fill=1.0)
    var out = _zeros(2)
    var scratch = _zeros(64)
    var q_proj = List[Float32](length=4, fill=1.0)
    var k_proj = List[Float32](length=2, fill=1.0)
    var v_proj = List[Float32](length=2, fill=1.0)
    var o_proj = List[Float32](length=4, fill=1.0)
    var weights = LayerWeights()
    weights.q_proj = TensorInfo(Int(q_proj.unsafe_ptr()), 2, 2)
    weights.k_proj = TensorInfo(Int(k_proj.unsafe_ptr()), 1, 2)
    weights.v_proj = TensorInfo(Int(v_proj.unsafe_ptr()), 1, 2)
    weights.o_proj = TensorInfo(Int(o_proj.unsafe_ptr()), 2, 2)
    var backend = TrackingBackend()

    forward_full_attention(
        backend,
        _ptr(out),
        _ptr(x),
        weights,
        0,
        0,
        2,
        2,
        1,
        1,
        kv_cache,
        rope_tables,
        False,
        4,
        attn_logit_softcapping,
        _ptr(scratch),
    )

    _ = layer_types
    _ = x[0]
    _ = out[0]
    _ = scratch[0]
    _ = q_proj[0]
    _ = k_proj[0]
    _ = v_proj[0]
    _ = o_proj[0]
    return backend^


def _call_moe_attention(attn_logit_softcapping: Float32) raises -> TrackingBackend:
    var layer_types = List[UInt8](length=1, fill=UInt8(LAYER_TYPE_SLIDING))
    var kv_cache = KVCache(
        1,
        1,
        2,
        4,
        4,
        UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(layer_types.unsafe_ptr())),
    )
    var rope_tables = RoPETables(2, 1.0, 4, 4)
    var x = List[Float32](length=2, fill=1.0)
    var out = _zeros(2)
    var scratch = _zeros(256)
    var qkv = List[Float32](length=4, fill=1.0)
    var o_proj = List[Float32](length=4, fill=1.0)
    var norm2 = List[Float32](length=2, fill=0.0)
    var dense_gate = List[Float32](length=2, fill=1.0)
    var dense_up = List[Float32](length=2, fill=1.0)
    var dense_down = List[Float32](length=2, fill=1.0)
    var router_proj = List[Float32](length=2, fill=1.0)
    var router_scale = List[Float32](length=2, fill=1.0)
    var per_expert_scale = List[Float32](length=1, fill=1.0)
    var expert_gate_up = List[Float32](length=4, fill=1.0)
    var expert_down = List[Float32](length=2, fill=1.0)
    var weights = MoELayerWeights()
    weights.q_proj = TensorInfo(Int(qkv.unsafe_ptr()), 2, 2)
    weights.k_proj = TensorInfo(Int(qkv.unsafe_ptr()), 2, 2)
    weights.v_proj = TensorInfo(Int(qkv.unsafe_ptr()), 2, 2)
    weights.o_proj = TensorInfo(Int(o_proj.unsafe_ptr()), 2, 2)
    weights.input_layernorm = TensorInfo(Int(norm2.unsafe_ptr()), 2, 1)
    weights.post_attention_layernorm = TensorInfo(Int(norm2.unsafe_ptr()), 2, 1)
    weights.pre_feedforward_layernorm = TensorInfo(Int(norm2.unsafe_ptr()), 2, 1)
    weights.dense_gate_proj = TensorInfo(Int(dense_gate.unsafe_ptr()), 1, 2)
    weights.dense_up_proj = TensorInfo(Int(dense_up.unsafe_ptr()), 1, 2)
    weights.dense_down_proj = TensorInfo(Int(dense_down.unsafe_ptr()), 2, 1)
    weights.post_feedforward_layernorm_1 = TensorInfo(Int(norm2.unsafe_ptr()), 2, 1)
    weights.pre_feedforward_layernorm_2 = TensorInfo(Int(norm2.unsafe_ptr()), 2, 1)
    weights.router_proj = TensorInfo(Int(router_proj.unsafe_ptr()), 1, 2)
    weights.router_scale = TensorInfo(Int(router_scale.unsafe_ptr()), 2, 1)
    weights.per_expert_scale = TensorInfo(Int(per_expert_scale.unsafe_ptr()), 1, 1)
    weights.expert_gate_up_proj = TensorInfo(Int(expert_gate_up.unsafe_ptr()), 1, 4)
    weights.expert_down_proj = TensorInfo(Int(expert_down.unsafe_ptr()), 2, 1)
    weights.post_feedforward_layernorm_2 = TensorInfo(Int(norm2.unsafe_ptr()), 2, 1)
    var backend = TrackingBackend()
    var dummy_stage = 0
    var dummy_ctx = 0

    forward_moe_layer(
        backend,
        _ptr(out),
        _ptr(x),
        weights,
        0,
        0,
        2,
        1,
        1,
        2,
        1,
        1,
        1,
        kv_cache,
        rope_tables,
        4,
        attn_logit_softcapping,
        _ptr(scratch),
        dummy_stage,
        dummy_ctx,
    )

    _ = layer_types
    _ = x[0]
    _ = out[0]
    _ = scratch[0]
    _ = qkv[0]
    _ = o_proj[0]
    _ = norm2[0]
    _ = dense_gate[0]
    _ = dense_up[0]
    _ = dense_down[0]
    _ = router_proj[0]
    _ = router_scale[0]
    _ = per_expert_scale[0]
    _ = expert_gate_up[0]
    _ = expert_down[0]
    return backend^


def test_attention_softcap_runs_before_softmax() raises:
    var backend = _call_sliding_attention(50.0)
    _assert_true(backend.softcap_calls == 1, "expected attention softcap to run once")
    _assert_true(backend.softmax_calls == 2, "expected per-head softmax calls")
    _assert_true(
        backend.softmax_calls_at_first_softcap == 0,
        "expected sliding attention softcap before any softmax",
    )
    _assert_true(
        backend.softcap_before_softmax,
        "expected attention softcap to run before softmax",
    )


def test_attention_softcap_disabled_at_zero() raises:
    var backend = _call_sliding_attention(0.0)
    _assert_true(backend.softcap_calls == 0, "softcap cap <= 0 should be disabled")
    _assert_true(backend.softmax_calls == 2, "softmax should still run when softcap is disabled")


def test_full_attention_softcap_runs_before_softmax() raises:
    var backend = _call_full_attention(50.0)
    _assert_true(backend.softcap_calls == 1, "expected full attention softcap to run once")
    _assert_true(backend.softmax_calls == 2, "expected per-head full attention softmax calls")
    _assert_true(
        backend.softmax_calls_at_first_softcap == 0,
        "expected full attention softcap before any softmax",
    )
    _assert_true(
        backend.softcap_before_softmax,
        "expected full attention softcap to run before softmax",
    )


def test_moe_attention_softcap_runs_before_softmax() raises:
    var backend = _call_moe_attention(50.0)
    _assert_true(backend.softcap_calls == 1, "expected MoE attention softcap to run once")
    _assert_true(
        backend.softmax_calls_at_first_softcap == 0,
        "expected MoE attention softcap before attention and router softmax",
    )
    _assert_true(backend.softmax_calls >= 2, "expected MoE attention and router softmax calls")


def test_final_logit_softcap() raises:
    var embed = List[Float32](length=1, fill=100.0)
    var norm = List[Float32](length=1, fill=0.0)
    var lm_head = List[Float32](length=1, fill=100.0)
    var model = ModelWeights()
    model.embed_tokens = TensorInfo(Int(embed.unsafe_ptr()), 1, 1)
    model.norm = TensorInfo(Int(norm.unsafe_ptr()), 1, 1)
    model.lm_head = TensorInfo(Int(lm_head.unsafe_ptr()), 1, 1)
    model.layers = []

    var layer_types = List[UInt8](length=1, fill=UInt8(LAYER_TYPE_SLIDING))
    var kv_cache = KVCache(
        0,
        1,
        1,
        4,
        4,
        UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(layer_types.unsafe_ptr())),
    )
    var rope_tables = RoPETables(1, 1.0, 4, 4)
    var capped_logits = _zeros(1)
    var raw_logits = _zeros(1)
    var scratch = _zeros(16)
    var backend = CPUBackend()
    var dummy_stage = 0
    var dummy_ctx = 0
    var dummy_persistent = 0

    forward_gemma4_step(
        backend,
        _ptr(capped_logits),
        0,
        0,
        model,
        1,
        1,
        1,
        1,
        1,
        1,
        kv_cache,
        rope_tables,
        False,
        4,
        0.0,
        30.0,
        _ptr(scratch),
        dummy_stage,
        dummy_ctx,
        dummy_persistent,
    )

    forward_gemma4_step(
        backend,
        _ptr(raw_logits),
        0,
        0,
        model,
        1,
        1,
        1,
        1,
        1,
        1,
        kv_cache,
        rope_tables,
        False,
        4,
        0.0,
        0.0,
        _ptr(scratch),
        dummy_stage,
        dummy_ctx,
        dummy_persistent,
    )

    assert_almost_equal(capped_logits[0], 29.9237, atol=1e-3)
    _assert_true(raw_logits[0] > 30.0, "uncapped final logit should remain above cap")

    _ = embed[0]
    _ = norm[0]
    _ = lm_head[0]
    _ = layer_types
    _ = scratch[0]


def main() raises:
    test_attention_softcap_runs_before_softmax()
    test_attention_softcap_disabled_at_zero()
    test_full_attention_softcap_runs_before_softmax()
    test_moe_attention_softcap_runs_before_softmax()
    test_final_logit_softcap()
    print("test_softcap.mojo passed!")
