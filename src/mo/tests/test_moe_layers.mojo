from std.collections import List
from std.memory import UnsafePointer

from mogemma.layers import forward_moe_experts, forward_moe_router
from mogemma.model import TensorInfo
from mogemma.ops import CPUBackend


def _ptr(ref values: List[Float32]) -> UnsafePointer[Float32, MutExternalOrigin]:
    return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(values.unsafe_ptr()))


def _i32_ptr(ref values: List[Int32]) -> UnsafePointer[Int32, MutExternalOrigin]:
    return UnsafePointer[Int32, MutExternalOrigin](unsafe_from_address=Int(values.unsafe_ptr()))


def test_forward_moe_router_matches_reference() raises:
    var router_proj_values = List[Float32](length=6, fill=0.0)
    # [ [1, 0], [0, 1], [1, 1] ]
    router_proj_values[0] = 1.0
    router_proj_values[3] = 1.0
    router_proj_values[4] = 1.0
    router_proj_values[5] = 1.0

    var router_scale_values = List[Float32](length=2, fill=1.0)
    var per_expert_scale_values = List[Float32](length=3, fill=0.0)
    per_expert_scale_values[0] = 1.0
    per_expert_scale_values[1] = 2.0
    per_expert_scale_values[2] = 3.0
    var hidden_values = List[Float32](length=2, fill=0.0)
    hidden_values[0] = 3.0
    hidden_values[1] = 4.0
    var scratch = List[Float32](length=8, fill=0.0)
    var expert_indices = List[Int32](length=2, fill=0)
    var expert_weights = List[Float32](length=2, fill=0.0)

    var backend = CPUBackend()
    var router_proj = TensorInfo(Int(_ptr(router_proj_values)), 3, 2)
    var router_scale = TensorInfo(Int(_ptr(router_scale_values)), 2, 1)
    var per_expert_scale = TensorInfo(Int(_ptr(per_expert_scale_values)), 3, 1)

    forward_moe_router(
        backend,
        _i32_ptr(expert_indices),
        _ptr(expert_weights),
        _ptr(hidden_values),
        router_proj,
        router_scale,
        per_expert_scale,
        2,
        3,
        2,
        _ptr(scratch),
    )

    if expert_indices[0] != 2 or expert_indices[1] != 1:
        raise Error("router should pick experts 2 and 1 in descending order")

    var diff0 = expert_weights[0] - 1.9347339
    if diff0 < 0.0:
        diff0 = -diff0
    if diff0 > 5e-3:
        raise Error("top-1 router weight mismatch")

    var diff1 = expert_weights[1] - 0.7101770
    if diff1 < 0.0:
        diff1 = -diff1
    if diff1 > 5e-3:
        raise Error("top-2 router weight mismatch")


def test_forward_moe_experts_packed_matches_reference() raises:
    var expert_gate_up_values = List[Float32](length=8, fill=0.0)
    # Expert 0: gate=[1,0], up=[0,1]
    expert_gate_up_values[0] = 1.0
    expert_gate_up_values[3] = 1.0
    # Expert 1: gate=[0,1], up=[1,0]
    expert_gate_up_values[5] = 1.0
    expert_gate_up_values[6] = 1.0

    var expert_down_values = List[Float32](length=4, fill=0.0)
    expert_down_values[0] = 1.0
    expert_down_values[1] = 2.0
    expert_down_values[2] = 3.0
    expert_down_values[3] = 4.0

    var hidden_values = List[Float32](length=2, fill=0.0)
    hidden_values[0] = 1.0
    hidden_values[1] = 2.0

    var out = List[Float32](length=2, fill=0.0)
    var scratch = List[Float32](length=8, fill=0.0)
    var expert_indices = List[Int32](length=2, fill=0)
    expert_indices[0] = 0
    expert_indices[1] = 1
    var expert_weights = List[Float32](length=2, fill=0.0)
    expert_weights[0] = 0.25
    expert_weights[1] = 0.75

    var backend = CPUBackend()
    var expert_gate_up = TensorInfo(Int(_ptr(expert_gate_up_values)), 2, 4)
    var expert_down = TensorInfo(Int(_ptr(expert_down_values)), 2, 2)
    var dummy_stage = 0
    var dummy_ctx = 0

    forward_moe_experts(
        backend,
        _ptr(out),
        _ptr(hidden_values),
        _i32_ptr(expert_indices),
        _ptr(expert_weights),
        expert_gate_up,
        expert_down,
        2,
        2,
        1,
        _ptr(scratch),
        dummy_stage,
        dummy_ctx,
    )

    var diff0 = out[0] - 4.8182507
    if diff0 < 0.0:
        diff0 = -diff0
    if diff0 > 5e-3:
        raise Error("packed expert output[0] mismatch")

    var diff1 = out[1] - 6.7056289
    if diff1 < 0.0:
        diff1 = -diff1
    if diff1 > 5e-3:
        raise Error("packed expert output[1] mismatch")


def main() raises:
    test_forward_moe_router_matches_reference()
    test_forward_moe_experts_packed_matches_reference()
    print("test_moe_layers.mojo passed!")
