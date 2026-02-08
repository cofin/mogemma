from testing import assert_equal
from python import Python, PythonObject
from sampling import Sampler

def test_sampling_greedy():
    var np = Python.import_module("numpy")
    var list = Python.list()
    list.append(0.1)
    list.append(0.5)
    list.append(0.4)
    var logits = np.array(list, dtype=np.float32)
    var sampler = Sampler()
    var token = sampler.greedy(logits)
    assert_equal(token, 1)

def main():
    try:
        test_sampling_greedy()
        print("Test Passed")
    except e:
        print("Test Failed:", e)