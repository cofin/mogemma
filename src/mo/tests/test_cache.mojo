from testing import assert_equal
from cache import KVCache

def test_cache_init():
    var cache = KVCache(num_layers=2, batch_size=1, max_seq_len=128, head_dim=64, num_heads=8)
    assert_equal(cache.num_layers, 2)
    assert_equal(cache.batch_size, 1)
    assert_equal(cache.max_seq_len, 128)
    # Check if buffer is allocated (placeholder for now)
    assert_equal(cache.is_allocated(), True)

def main():
    try:
        test_cache_init()
        print("Test Passed")
    except e:
        print("Test Failed:", e)
