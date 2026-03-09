from memory import UnsafePointer, alloc
from mogemma.model import VisionLayerWeights
from mogemma.layers import forward_vision_attention, forward_vision_mlp, forward_vision_layer


fn test_forward_vision_layer():
    var num_patches = 2
    var hidden_size = 4
    var num_heads = 2
    var head_dim = 2
    var intermediate_size = 8

    var weights = VisionLayerWeights()
    # just testing compilation and basic execution without crashing

    var x_ptr = alloc[Float32](num_patches * hidden_size)
    var out_ptr = alloc[Float32](num_patches * hidden_size)
    var scratch_ptr = alloc[Float32](num_patches * hidden_size * 10)

    for i in range(num_patches * hidden_size):
        x_ptr.store(i, 1.0)

    forward_vision_layer(
        out_ptr,
        x_ptr,
        weights,
        num_patches,
        hidden_size,
        num_heads,
        head_dim,
        intermediate_size,
        scratch_ptr,
    )

    # Just check it completed
    var val = out_ptr.load(0)
    if val < -1000.0 or val > 1000.0:
        print("Out of bounds")

    x_ptr.free()
    out_ptr.free()
    scratch_ptr.free()


fn main():
    test_forward_vision_layer()
    print("test_vision_layers passed")
