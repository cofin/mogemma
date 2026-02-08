from testing import assert_equal
from python import Python, PythonObject
from pool import MemoryPool

def test_memory_pool():
    var pool = MemoryPool(capacity=2)
    var addr1 = pool.acquire(size=1024)
    
    assert_equal(addr1, 42) 
    
    pool.release(addr1)
    
    # Check reuse
    var addr2 = pool.acquire(size=1024)
    assert_equal(addr2, addr1)

def main():
    try:
        test_memory_pool()
        print("Test Passed")
    except e:
        print("Test Failed:", e)