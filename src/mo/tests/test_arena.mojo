from std.memory import *
from mogemma.core import MemoryArena
from std.testing import assert_equal


def test_arena_allocation() raises:
    var size = 1024
    var arena = MemoryArena(size)
    assert_equal(arena.size, size)

    # Simple check that pointer is non-null
    var ptr_addr = Int(arena.ptr)
    if ptr_addr == 0:
        raise Error("Arena pointer should not be null")

    # Test writing and reading
    arena.ptr.store(0, 1.234)
    arena.ptr.store(size - 1, 5.678)

    assert_equal(arena.ptr.load(0), Float32(1.234))
    assert_equal(arena.ptr.load(size - 1), Float32(5.678))

    arena.free()
    print("test_arena_allocation passed")


def main():
    try:
        test_arena_allocation()
    except e:
        print("Test failed:", e)
