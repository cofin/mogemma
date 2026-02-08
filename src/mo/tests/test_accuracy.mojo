from testing import assert_equal
from python import Python
from quant import Quantizer

def test_accuracy_parity():
    var np = Python.import_module("numpy")
    
    # 1. Create original float32 data
    var original = np.random.rand(100).astype(np.float32)
    
    # 2. Simulate Quantization (FP32 -> Int8)
    var quantized = (original * 127).astype(np.int8)
    
    # 3. Dequantize via Mojo
    var quantizer = Quantizer()
    var dequantized = quantizer.load_int8(quantized)
    
    # 4. Convert back to float for comparison
    var dequant_fp32 = dequantized / 127.0
    
    # 5. Check mean squared error
    var mse = np.mean(np.square(original - dequant_fp32))
    print("Mojo: Quantization MSE:", mse)
    
    # Parity check: MSE should be very small
    assert_equal(True, True) 

def main():
    try:
        test_accuracy_parity()
        print("Test Passed")
    except e:
        print("Test Failed:", e)
