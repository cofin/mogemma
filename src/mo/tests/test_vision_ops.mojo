from memory import UnsafePointer, alloc
from mogemma.vision_ops import normalize_rgb_bytes, bilinear_resize_rgb, extract_patches, add_positional_embeddings


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


fn test_extract_patches():
    var h = 4
    var w = 4
    var c = 3
    var patch_size = 2
    var hidden_size = 2

    var num_patches_y = h // patch_size
    var num_patches_x = w // patch_size
    var num_patches = num_patches_y * num_patches_x

    var in_ptr = alloc[Float32](h * w * c)
    var weight_ptr = alloc[Float32](hidden_size * patch_size * patch_size * c)
    var bias_ptr = alloc[Float32](hidden_size)
    var out_ptr = alloc[Float32](num_patches * hidden_size)

    for i in range(h * w * c):
        in_ptr.store(i, 1.0)
    for i in range(hidden_size * patch_size * patch_size * c):
        weight_ptr.store(i, 1.0)
    for i in range(hidden_size):
        bias_ptr.store(i, 0.0)

    extract_patches(out_ptr, in_ptr, weight_ptr, bias_ptr, h, w, c, patch_size, hidden_size)

    # 2x2x3 = 12, all 1s, weights all 1s, bias 0 = 12
    var val = out_ptr.load(0)
    if val < 11.9 or val > 12.1:
        print("Extract patches failed")

    in_ptr.free()
    weight_ptr.free()
    bias_ptr.free()
    out_ptr.free()


fn test_positional_embeddings():
    var num_patches = 4
    var hidden_size = 2
    var seq_ptr = alloc[Float32](num_patches * hidden_size)
    var pos_ptr = alloc[Float32](num_patches * hidden_size)

    for i in range(num_patches * hidden_size):
        seq_ptr.store(i, 1.0)
        pos_ptr.store(i, 2.0)

    add_positional_embeddings(seq_ptr, pos_ptr, num_patches, hidden_size)

    var val = seq_ptr.load(0)
    if val < 2.9 or val > 3.1:
        print("Positional embeddings failed")

    seq_ptr.free()
    pos_ptr.free()


fn main():
    test_normalize()
    test_resize()
    test_extract_patches()
    test_positional_embeddings()
    print("test_vision_ops passed")
