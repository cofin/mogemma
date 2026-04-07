from mogemma.model import ModelWeights, LayerWeights, KVCache, RoPETables, LAYER_TYPE_SLIDING, LAYER_TYPE_FULL
from std.memory import UnsafePointer
from std.collections import List


def main():
    var m = ModelWeights()
    var layer = LayerWeights()
    m.layers.append(layer^)
    print("Model initialized. Layer count:", len(m.layers))

    # Test KVCache construction
    var layer_types = List[UInt8](length=2, fill=UInt8(0))
    layer_types[0] = LAYER_TYPE_SLIDING
    layer_types[1] = LAYER_TYPE_FULL
    var lt_ptr = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(layer_types.unsafe_ptr()))
    var cache = KVCache(2, 4, 32, 512, 4096, lt_ptr)
    print("KVCache initialized. Total elements:", cache.total_elements())

    # Test RoPETables construction
    var rope = RoPETables(32, 0.5, 512, 4096)
    print("RoPETables initialized. rotary_dim:", rope.rotary_dim)
    _ = layer_types
