from std.python import Python, PythonObject
from mogemma.core import init_model_mojo, free_arena_mojo
from std.testing import assert_equal


def wrap_tensor(arr: PythonObject) raises -> PythonObject:
    return Python.tuple(Int(py=arr.__array_interface__["data"][0]), arr.shape, "float32")


def test_init_with_arena() raises:
    var np = Python.import_module("numpy")
    var metadata = Python.dict()
    # Minimal metadata for standard model
    var arr = np.zeros(Python.tuple(100, 128), dtype=np.float32)
    metadata["model.embed_tokens.weight"] = Python.tuple(
        Int(py=arr.__array_interface__["data"][0]), arr.shape, "float32"
    )
    metadata["model.norm.weight"] = wrap_tensor(np.zeros(Python.tuple(128), dtype=np.float32))
    metadata["lm_head.weight"] = wrap_tensor(np.zeros(Python.tuple(100, 128), dtype=np.float32))

    # Layers
    var pfx = "model.layers.0"
    metadata[pfx + ".input_layernorm.weight"] = wrap_tensor(np.zeros(Python.tuple(128), dtype=np.float32))
    metadata[pfx + ".post_attention_layernorm.weight"] = wrap_tensor(np.zeros(Python.tuple(128), dtype=np.float32))
    metadata[pfx + ".self_attn.q_proj.weight"] = wrap_tensor(np.zeros(Python.tuple(128, 128), dtype=np.float32))
    metadata[pfx + ".self_attn.k_proj.weight"] = wrap_tensor(np.zeros(Python.tuple(128, 128), dtype=np.float32))
    metadata[pfx + ".self_attn.v_proj.weight"] = wrap_tensor(np.zeros(Python.tuple(128, 128), dtype=np.float32))
    metadata[pfx + ".self_attn.o_proj.weight"] = wrap_tensor(np.zeros(Python.tuple(128, 128), dtype=np.float32))
    metadata[pfx + ".mlp.gate_proj.weight"] = wrap_tensor(np.zeros(Python.tuple(256, 128), dtype=np.float32))
    metadata[pfx + ".mlp.up_proj.weight"] = wrap_tensor(np.zeros(Python.tuple(256, 128), dtype=np.float32))
    metadata[pfx + ".mlp.down_proj.weight"] = wrap_tensor(np.zeros(Python.tuple(128, 256), dtype=np.float32))
    metadata[pfx + ".self_attn.q_norm.weight"] = wrap_tensor(np.zeros(Python.tuple(128), dtype=np.float32))
    metadata[pfx + ".self_attn.k_norm.weight"] = wrap_tensor(np.zeros(Python.tuple(128), dtype=np.float32))
    metadata[pfx + ".pre_feedforward_layernorm.weight"] = wrap_tensor(np.zeros(Python.tuple(128), dtype=np.float32))
    metadata[pfx + ".post_feedforward_layernorm.weight"] = wrap_tensor(np.zeros(Python.tuple(128), dtype=np.float32))

    var llm = init_model_mojo(metadata)

    var arena_ptr = Int(py=llm["_arena_ptr"])
    var arena_size = Int(py=llm["_arena_size"])

    if arena_ptr == 0:
        raise Error("_arena_ptr should not be zero")
    if arena_size <= 0:
        raise Error("_arena_size should be positive")

    print("Arena initialized at", arena_ptr, "with size", arena_size)

    # Check that individual buffers are offsets in the arena
    var k_cache_ptr = Int(py=llm["k_cache"])
    var v_cache_ptr = Int(py=llm["v_cache"])

    if k_cache_ptr < arena_ptr or k_cache_ptr >= arena_ptr + arena_size * 4:
        raise Error("k_cache_ptr out of arena bounds")

    free_arena_mojo(llm)
    assert_equal(Int(py=llm["_arena_ptr"]), 0)
    print("test_init_with_arena passed")


def main():
    try:
        test_init_with_arena()
    except e:
        print("Test failed:", e)
