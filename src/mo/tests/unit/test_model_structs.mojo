from std.memory import alloc
from std.testing import assert_equal

from mogemma.core import MemoryArena
from mogemma.model import (
    KVCache,
    LAYER_TYPE_FULL,
    LAYER_TYPE_SLIDING,
    LayerWeights,
    ModelWeights,
    RoPETables,
)


def test_memory_arena_allocation() raises:
    var size = 1024
    var arena = MemoryArena(size)
    assert_equal(arena.size, size)

    if Int(arena.ptr) == 0:
        raise Error("Arena pointer should not be null")

    arena.ptr.store(0, 1.234)
    arena.ptr.store(size - 1, 5.678)

    assert_equal(arena.ptr.load(0), Float32(1.234))
    assert_equal(arena.ptr.load(size - 1), Float32(5.678))

    arena.free()
    print("test_memory_arena_allocation passed")


def test_model_weights_layer_append() raises:
    var weights = ModelWeights()
    var layer = LayerWeights()
    weights.layers.append(layer^)

    assert_equal(len(weights.layers), 1)
    print("test_model_weights_layer_append passed")


def test_kv_cache_hybrid_layout() raises:
    var layer_types_ptr = alloc[UInt8](2)
    layer_types_ptr.store(0, LAYER_TYPE_SLIDING)
    layer_types_ptr.store(1, LAYER_TYPE_FULL)

    var cache = KVCache(2, 4, 32, 512, 4096, layer_types_ptr)

    assert_equal(cache.get_layer_type(0), LAYER_TYPE_SLIDING)
    assert_equal(cache.get_layer_type(1), LAYER_TYPE_FULL)
    assert_equal(cache.get_layer_cache_size(0), 512)
    assert_equal(cache.get_layer_cache_size(1), 4096)
    assert_equal(cache.get_layer_offset(0), 0)
    assert_equal(cache.get_layer_offset(1), 512 * 4 * 32)
    assert_equal(cache.total_elements(), (512 + 4096) * 4 * 32)
    layer_types_ptr.free()
    print("test_kv_cache_hybrid_layout passed")


def test_rope_tables_construction() raises:
    var rope = RoPETables(32, 0.5, 512, 4096)
    assert_equal(rope.head_dim, 32)
    assert_equal(rope.rotary_dim, 16)
    assert_equal(len(rope.sliding_cos), 512 * 16)
    assert_equal(len(rope.sliding_sin), 512 * 16)
    assert_equal(len(rope.full_cos), 4096 * 8)
    assert_equal(len(rope.full_sin), 4096 * 8)
    print("test_rope_tables_construction passed")


def main() raises:
    test_memory_arena_allocation()
    test_model_weights_layer_append()
    test_kv_cache_hybrid_layout()
    test_rope_tables_construction()
    print("test_model_structs.mojo passed!")
