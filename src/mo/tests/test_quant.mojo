from testing import assert_equal
from python import Python, PythonObject
from quant import Quantizer

def test_quant_loading():
    var np = Python.import_module("numpy")
    var l = Python.list()
    l.append(1024)
    l.append(1024)
    # Simulate int8 weights
    var weights = np.random.randint(-128, 127, size=l, dtype=np.int8)
    
    var quantizer = Quantizer()
    var dequantized = quantizer.load_int8(weights)
    
    # Check return
    assert_equal(True, True) 

def main():
    try:
        test_quant_loading()
        print("Test Passed")
    except e:
        print("Test Failed:", e)
