from mogemma.model import ModelWeights, LayerWeights, VisionModelWeights, VisionLayerWeights


fn main():
    var m = ModelWeights()
    var layer = LayerWeights()
    m.layers.append(layer^)
    print("Model initialized. Layer count:", len(m.layers))

    var vm = VisionModelWeights()
    var vlayer = VisionLayerWeights()
    vm.layers.append(vlayer^)
    print("Vision Model initialized. Layer count:", len(vm.layers))
