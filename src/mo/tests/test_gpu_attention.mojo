"""Tests for GPU attention kernels in ops_gpu.mojo."""

from std.sys import has_accelerator
from std.memory import UnsafePointer
from std.collections import List
from std.testing import assert_almost_equal

from mogemma.ops_gpu import (
    _kv_write_impl,
    _attention_scores_impl,
    _attention_value_accum_impl,
)


def test_kv_write_kernel_logic() raises:
    """Verify kv_write_kernel logic using CPU pointers (as kernels are just functions)."""
    var kv_size = 4
    var cache_size = 2
    var layer_offset = 0

    var src: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var dst = List[Float32](length=8, fill=0.0)

    var src_ptr = src.unsafe_ptr()
    var dst_ptr = dst.unsafe_ptr()

    # Task 3.1: sliding window write (ring buffer)
    # pos=0 -> write_pos=0
    for tid in range(kv_size):
        _kv_write_impl(tid, dst_ptr, src_ptr, kv_size, 0, cache_size, layer_offset, False)
    for i in range(kv_size):
        if dst[i] != src[i]:
            raise Error("Sliding write at pos=0 failed")

    # pos=2 -> write_pos=0 (wrap)
    src[0] = 5.0
    for tid in range(kv_size):
        _kv_write_impl(tid, dst_ptr, src_ptr, kv_size, 2, cache_size, layer_offset, False)
    if dst[0] != 5.0:
        raise Error("Sliding write at pos=2 (wrap) failed")

    # pos=1 -> write_pos=1
    src[0] = 6.0
    for tid in range(kv_size):
        _kv_write_impl(tid, dst_ptr, src_ptr, kv_size, 1, cache_size, layer_offset, False)
    if dst[4] != 6.0:
        raise Error("Sliding write at pos=1 failed")

    # Task 3.1: full attention write (linear)
    # pos=2 -> write_pos=2 (no wrap)
    var dst_full = List[Float32](length=12, fill=0.0)
    var dst_full_ptr = dst_full.unsafe_ptr()
    src[0] = 7.0
    for tid in range(kv_size):
        _kv_write_impl(tid, dst_full_ptr, src_ptr, kv_size, 2, 100, layer_offset, True)
    if dst_full[8] != 7.0:
        raise Error("Full write at pos=2 failed")

    print("  test_kv_write_kernel_logic passed")


def test_attention_scores_kernel_logic() raises:
    """Verify _attention_scores_impl logic using CPU pointers."""
    var num_heads = 2
    var num_kv_heads = 1
    var head_dim = 2
    var valid_len = 2
    var kv_size = num_kv_heads * head_dim  # 2
    var scale: Float32 = 1.0

    # Q: [2 heads, 2 dim] -> [1, 0, 0, 1]
    var q: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    # K cache: [2 slots, 2 dim] -> [[1, 0], [0, 1]]
    var k_cache: List[Float32] = [1.0, 0.0, 0.0, 1.0]
    # Output: [2 heads, 2 valid]
    var scores = List[Float32](length=4, fill=0.0)

    var q_ptr = q.unsafe_ptr()
    var k_ptr = k_cache.unsafe_ptr()
    var s_ptr = scores.unsafe_ptr()

    for h in range(num_heads):
        for tid in range(valid_len):
            _attention_scores_impl(
                h,
                tid,
                valid_len,
                s_ptr,
                q_ptr,
                k_ptr,
                num_heads,
                num_kv_heads,
                head_dim,
                valid_len,
                kv_size,
                scale,
            )

    if scores[0] != 1.0 or scores[1] != 0.0:
        raise Error("Head 0 scores incorrect")
    if scores[2] != 0.0 or scores[3] != 1.0:
        raise Error("Head 1 scores incorrect")

    print("  test_attention_scores_kernel_logic passed")


def test_attention_value_accum_kernel_logic() raises:
    """Verify _attention_value_accum_impl logic using CPU pointers."""
    var num_heads = 2
    var num_kv_heads = 1
    var head_dim = 2
    var valid_len = 2
    var kv_size = num_kv_heads * head_dim  # 2

    # Scores (after softmax): [2 heads, 2 valid]
    # Head 0: [1, 0] (attend fully to first token)
    # Head 1: [0.5, 0.5] (average both tokens)
    var scores: List[Float32] = [1.0, 0.0, 0.5, 0.5]

    # V cache: [2 slots, 2 dim] -> [[10, 20], [30, 40]]
    var v_cache: List[Float32] = [10.0, 20.0, 30.0, 40.0]

    # Output: [2 heads, 2 dim]
    var out = List[Float32](length=4, fill=0.0)

    var s_ptr = scores.unsafe_ptr()
    var v_ptr = v_cache.unsafe_ptr()
    var o_ptr = out.unsafe_ptr()

    for h in range(num_heads):
        for tid in range(head_dim):
            _attention_value_accum_impl(
                h,
                tid,
                head_dim,
                o_ptr,
                s_ptr,
                v_ptr,
                num_heads,
                num_kv_heads,
                head_dim,
                valid_len,
                kv_size,
            )

    # Head 0: 1.0 * [10, 20] + 0.0 * [30, 40] = [10, 20]
    if out[0] != 10.0 or out[1] != 20.0:
        raise Error("Head 0 accumulation incorrect")
    # Head 1: 0.5 * [10, 20] + 0.5 * [30, 40] = [20, 30]
    if out[2] != 20.0 or out[3] != 30.0:
        raise Error("Head 1 accumulation incorrect")

    print("  test_attention_value_accum_kernel_logic passed")


def main() raises:
    test_kv_write_kernel_logic()
    test_attention_scores_kernel_logic()
    test_attention_value_accum_kernel_logic()
