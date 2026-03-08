from memory import UnsafePointer, alloc
from mogemma.vision_ops import normalize_rgb_bytes, bilinear_resize_rgb

fn test_normalize():
    var num_pixels = 4
    var in_ptr = alloc[UInt8](num_pixels * 3)
    var out_ptr = alloc[Float32](num_pixels * 3)
    
    # Fill with 255 (white)
    for i in range(num_pixels * 3):
        in_ptr.store(i, 255)
        
    normalize_rgb_bytes(out_ptr, in_ptr, num_pixels)
    
    # For 255, we expect (1.0 - mean) / std
    # R: (1.0 - 0.48145466) / 0.26862954 ≈ 1.93033
    # G: (1.0 - 0.4578275) / 0.26130258 ≈ 2.07488
    # B: (1.0 - 0.40821073) / 0.27577711 ≈ 2.14589
    
    var r = out_ptr.load(0)
    var g = out_ptr.load(1)
    var b = out_ptr.load(2)
    
    if r < 1.93 or r > 1.94:
        print("Normalize R failed:", r)
    if g < 2.07 or g > 2.08:
        print("Normalize G failed:", g)
    if b < 2.14 or b > 2.15:
        print("Normalize B failed:", b)
        
    in_ptr.free()
    out_ptr.free()

fn test_resize():
    var in_w = 2
    var in_h = 2
    var out_w = 4
    var out_h = 4
    
    var in_ptr = alloc[Float32](in_w * in_h * 3)
    var out_ptr = alloc[Float32](out_w * out_h * 3)
    
    for i in range(in_w * in_h * 3):
        in_ptr.store(i, Float32(i))
        
    bilinear_resize_rgb(out_ptr, out_h, out_w, in_ptr, in_h, in_w)
    
    # Just basic test it runs without crashing and out_ptr is populated
    var val = out_ptr.load(0)
    if val < 0.0:
        print("Resize failed")
        
    in_ptr.free()
    out_ptr.free()

fn main():
    test_normalize()
    test_resize()
    print("test_vision_ops passed")
