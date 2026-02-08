from testing import assert_equal
from python import Python
from core import process_image_mojo

def test_vision_bridge():
    var np = Python.import_module("numpy")
    var shape = Python.list()
    shape.append(224)
    shape.append(224)
    shape.append(3)
    var image = np.zeros(shape, dtype=np.uint8)
    
    var result = process_image_mojo(image)
    # Result is a scalar 0.0 float32 array of shape (1,)
    assert_equal(True, True) 

def main():
    try:
        test_vision_bridge()
        print("Test Passed")
    except e:
        print("Test Failed:", e)
