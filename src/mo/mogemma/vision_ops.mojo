from memory import UnsafePointer

# Pre-computed ImageNet normalization constants
# mean = [0.48145466, 0.4578275, 0.40821073]
# std = [0.26862954, 0.26130258, 0.27577711]

@always_inline
fn normalize_rgb_bytes[
    nelts: Int = 1
](
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    in_ptr: UnsafePointer[UInt8, MutExternalOrigin],
    num_pixels: Int,
):
    """Normalizes RGB bytes (0-255) to float32 tensors using standard ImageNet mean and std.
    Input must be packed HWC (RGB). Output is HWC float32.
    """
    var mean_r: Float32 = 0.48145466
    var mean_g: Float32 = 0.4578275
    var mean_b: Float32 = 0.40821073
    
    var std_r: Float32 = 0.26862954
    var std_g: Float32 = 0.26130258
    var std_b: Float32 = 0.27577711
    
    for i in range(num_pixels):
        var r = Float32(in_ptr.load(i * 3)) / 255.0
        var g = Float32(in_ptr.load(i * 3 + 1)) / 255.0
        var b = Float32(in_ptr.load(i * 3 + 2)) / 255.0
        
        out_ptr.store(i * 3, (r - mean_r) / std_r)
        out_ptr.store(i * 3 + 1, (g - mean_g) / std_g)
        out_ptr.store(i * 3 + 2, (b - mean_b) / std_b)

@always_inline
fn bilinear_resize_rgb(
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    out_h: Int,
    out_w: Int,
    in_ptr: UnsafePointer[Float32, MutExternalOrigin],
    in_h: Int,
    in_w: Int,
):
    """Performs bilinear resizing of an HWC image tensor (float32).
    """
    var scale_x = Float32(in_w) / Float32(out_w)
    var scale_y = Float32(in_h) / Float32(out_h)

    for y in range(out_h):
        for x in range(out_w):
            var src_x = (Float32(x) + 0.5) * scale_x - 0.5
            var src_y = (Float32(y) + 0.5) * scale_y - 0.5
            
            var x_floor = Int(src_x)
            var y_floor = Int(src_y)
            
            var x1 = max(0, min(in_w - 1, x_floor))
            var y1 = max(0, min(in_h - 1, y_floor))
            var x2 = max(0, min(in_w - 1, x_floor + 1))
            var y2 = max(0, min(in_h - 1, y_floor + 1))
            
            var x_weight = src_x - Float32(x_floor)
            var y_weight = src_y - Float32(y_floor)
            
            # Bound weights if out of image
            if src_x < 0:
                x_weight = 0
            if src_y < 0:
                y_weight = 0
                
            var w11 = (1.0 - x_weight) * (1.0 - y_weight)
            var w12 = x_weight * (1.0 - y_weight)
            var w21 = (1.0 - x_weight) * y_weight
            var w22 = x_weight * y_weight
            
            for c in range(3):
                var val11 = in_ptr.load((y1 * in_w + x1) * 3 + c)
                var val12 = in_ptr.load((y1 * in_w + x2) * 3 + c)
                var val21 = in_ptr.load((y2 * in_w + x1) * 3 + c)
                var val22 = in_ptr.load((y2 * in_w + x2) * 3 + c)
                
                var out_val = val11 * w11 + val12 * w12 + val21 * w21 + val22 * w22
                out_ptr.store((y * out_w + x) * 3 + c, out_val)

