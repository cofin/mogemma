from testing import assert_equal
from python import Python, PythonObject
from quant import Quantizer

def test_quant_loading():
    var np = Python.import_module("numpy")
    
    # Create known int8 weights using a Python list to avoid type inference issues
    var data = Python.list()
    data.append(-128)
    data.append(-1)
    data.append(0)
    data.append(1)
    data.append(127)
    var weights = np.array(data, dtype=np.int8)
    
    var quantizer = Quantizer()
    var dequantized = quantizer.load_int8(weights)
    
    # Verify values match (int8 cast to float32 should be exact)
    for i in range(5):
        var expected = Float32(py=weights[i])
        var actual = Float32(py=dequantized[i])
        if expected != actual:
            print("Mismatch at index", i, ": expected", expected, "got", actual)
            assert_equal(False, True)
    
    print("Verification successful!")

def test_quant_int4():
    var np = Python.import_module("numpy")
    
    # Create packed int4 weights (0x12 -> 2, 1)
    var data = Python.list()
    data.append(0x12) # low=2, high=1
    data.append(0x34) # low=4, high=3
    var weights = np.array(data, dtype=np.uint8)
    
    var quantizer = Quantizer()
    var dequantized = quantizer.load_int4(weights)
    
    # Verify values
    assert_equal(Float32(py=dequantized[0]), 2.0)
    assert_equal(Float32(py=dequantized[1]), 1.0)
    assert_equal(Float32(py=dequantized[2]), 4.0)
    assert_equal(Float32(py=dequantized[3]), 3.0)
    
    print("Int4 Verification successful!")

def main():
    try:
        test_quant_loading()
        test_quant_int4()
        print("All Tests Passed")
    except e:
        print("Test Failed:", e)
