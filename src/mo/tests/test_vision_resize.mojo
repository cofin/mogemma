from std.memory import UnsafePointer, alloc
from std.testing import assert_almost_equal
from mogemma.core import _resize_bilinear_rgb

fn test_resize_2x2_to_1x1() raises:
    # 2x2 RGB image (interleaved)
    var in_h = 2
    var in_w = 2
    var in_ptr = alloc[UInt8](in_h * in_w * 3)
    
    # Pixel 0,0: black
    in_ptr.store(0, 0)
    in_ptr.store(1, 0)
    in_ptr.store(2, 0)
    # Pixel 0,1: white
    in_ptr.store(3, 255)
    in_ptr.store(4, 255)
    in_ptr.store(5, 255)
    # Pixel 1,0: white
    in_ptr.store(6, 255)
    in_ptr.store(7, 255)
    in_ptr.store(8, 255)
    # Pixel 1,1: black
    in_ptr.store(9, 0)
    in_ptr.store(10, 0)
    in_ptr.store(11, 0)

    var out_h = 1
    var out_w = 1
    var out_ptr = alloc[Float32](out_h * out_w * 3)

    _resize_bilinear_rgb(
        out_ptr,
        in_ptr,
        in_h,
        in_w,
        out_h,
        out_w
    )

    assert_almost_equal(out_ptr.load(0), Float32(0.5))
    assert_almost_equal(out_ptr.load(1), Float32(0.5))
    assert_almost_equal(out_ptr.load(2), Float32(0.5))

    in_ptr.free()
    out_ptr.free()
    print("test_resize_2x2_to_1x1 passed")

fn main():
    try:
        test_resize_2x2_to_1x1()
    except e:
        print("Test failed:", e)
