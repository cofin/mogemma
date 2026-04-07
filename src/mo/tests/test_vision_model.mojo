from mogemma.model import VisionLayerWeights, VisionModelWeights, TensorInfo
from std.collections import List


def test_vision_layer_weights_default_init() raises:
    var vl = VisionLayerWeights()
    # All tensors should be zero-initialized
    print("VisionLayerWeights default init: shape_0=", vl.q_proj.shape_0, "shape_1=", vl.q_proj.shape_1)
    if vl.q_proj.shape_0 != 0 or vl.q_proj.shape_1 != 0:
        raise Error("q_proj should be zero-initialized")
    if vl.fc1.shape_0 != 0 or vl.fc2.shape_0 != 0:
        raise Error("fc1/fc2 should be zero-initialized")
    if vl.layer_norm1.shape_0 != 0 or vl.layer_norm2.shape_0 != 0:
        raise Error("layer_norm1/layer_norm2 should be zero-initialized")


def test_vision_model_weights_default_init() raises:
    var vm = VisionModelWeights()
    if vm.patch_embedding.shape_0 != 0:
        raise Error("patch_embedding should be zero-initialized")
    if vm.position_embedding.shape_0 != 0:
        raise Error("position_embedding should be zero-initialized")
    if vm.post_norm.shape_0 != 0:
        raise Error("post_norm should be zero-initialized")
    if vm.projection.shape_0 != 0:
        raise Error("projection should be zero-initialized")
    if len(vm.layers) != 0:
        raise Error("layers should be empty")
    print("VisionModelWeights default init: OK")


def test_vision_model_weights_with_layers() raises:
    var vm = VisionModelWeights()
    vm.patch_embedding = TensorInfo(100, 1152, 768)
    vm.position_embedding = TensorInfo(200, 1120, 1152)
    vm.post_norm = TensorInfo(300, 1152, 0)
    vm.projection = TensorInfo(400, 3584, 1152)

    for i in range(3):
        var vl = VisionLayerWeights()
        vl.q_proj = TensorInfo(1000 + i * 100, 1152, 1152)
        vl.k_proj = TensorInfo(1001 + i * 100, 1152, 1152)
        vl.v_proj = TensorInfo(1002 + i * 100, 1152, 1152)
        vl.o_proj = TensorInfo(1003 + i * 100, 1152, 1152)
        vl.fc1 = TensorInfo(1004 + i * 100, 4304, 1152)
        vl.fc2 = TensorInfo(1005 + i * 100, 1152, 4304)
        vl.layer_norm1 = TensorInfo(1006 + i * 100, 1152, 0)
        vl.layer_norm2 = TensorInfo(1007 + i * 100, 1152, 0)
        vm.layers.append(vl^)

    if len(vm.layers) != 3:
        raise Error("Expected 3 layers, got " + String(len(vm.layers)))
    if vm.layers[0].q_proj.shape_0 != 1152:
        raise Error("Layer 0 q_proj shape mismatch")
    if vm.layers[2].fc1.shape_0 != 4304:
        raise Error("Layer 2 fc1 shape mismatch")
    print("VisionModelWeights with 3 layers: OK")


def test_vision_layer_field_count() raises:
    # VisionLayerWeights should have exactly 8 TensorInfo fields:
    # q_proj, k_proj, v_proj, o_proj, fc1, fc2, layer_norm1, layer_norm2
    var vl = VisionLayerWeights()
    vl.q_proj = TensorInfo(1, 10, 20)
    vl.k_proj = TensorInfo(2, 10, 20)
    vl.v_proj = TensorInfo(3, 10, 20)
    vl.o_proj = TensorInfo(4, 10, 20)
    vl.fc1 = TensorInfo(5, 30, 10)
    vl.fc2 = TensorInfo(6, 10, 30)
    vl.layer_norm1 = TensorInfo(7, 10, 0)
    vl.layer_norm2 = TensorInfo(8, 10, 0)

    if vl.q_proj.shape_0 != 10:
        raise Error("q_proj shape check failed")
    if vl.fc2.shape_1 != 30:
        raise Error("fc2 shape check failed")
    print("VisionLayerWeights field count check: OK")


def main() raises:
    test_vision_layer_weights_default_init()
    test_vision_model_weights_default_init()
    test_vision_model_weights_with_layers()
    test_vision_layer_field_count()
    print("All vision model struct tests passed!")
