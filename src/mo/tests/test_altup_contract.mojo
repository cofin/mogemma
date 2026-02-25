from testing import assert_almost_equal
from memory import UnsafePointer
from collections import List

from mogemma.model import AltUpWeights, TensorInfo
from mogemma.layers import forward_altup_predict, forward_altup_correct


fn alloc_zeros(size: Int) -> List[Float32]:
    return List[Float32](length=size, fill=0.0)


fn alloc_ones(size: Int) -> List[Float32]:
    return List[Float32](length=size, fill=1.0)


fn get_ptr(lst: List[Float32]) -> UnsafePointer[Float32, MutExternalOrigin]:
    return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(lst.unsafe_ptr()))


fn test_forward_altup_predict_correct_identity_contract() raises:
    var hidden_size = 4
    var num_modalities = 4

    var weights = AltUpWeights()
    var router = alloc_zeros(num_modalities * hidden_size)
    var router_norm = alloc_ones(hidden_size)
    var prediction_coefs = alloc_zeros(num_modalities * num_modalities * num_modalities)
    var correction_coefs = alloc_zeros(num_modalities * num_modalities)
    var output_scale = alloc_ones(hidden_size)

    weights.router = TensorInfo(Int(router.unsafe_ptr()), num_modalities, hidden_size)
    weights.router_norm = TensorInfo(Int(router_norm.unsafe_ptr()), hidden_size, 1)
    weights.prediction_coefs = TensorInfo(Int(prediction_coefs.unsafe_ptr()), num_modalities, num_modalities * num_modalities)
    weights.correction_coefs = TensorInfo(Int(correction_coefs.unsafe_ptr()), num_modalities, num_modalities)
    weights.output_scale = TensorInfo(Int(output_scale.unsafe_ptr()), hidden_size, 0)

    var streams = alloc_ones(num_modalities * hidden_size)
    var predictions = alloc_zeros(num_modalities * hidden_size)
    var corrected = alloc_zeros(num_modalities * hidden_size)
    var activated = alloc_ones(hidden_size)
    var scratch = alloc_zeros(hidden_size * 4)

    forward_altup_predict(
        get_ptr(predictions),
        get_ptr(streams),
        weights,
        hidden_size,
        num_modalities,
        get_ptr(scratch),
    )

    forward_altup_correct(
        get_ptr(corrected),
        get_ptr(predictions),
        get_ptr(activated),
        weights,
        hidden_size,
        num_modalities,
        get_ptr(scratch),
    )

    for i in range(num_modalities * hidden_size):
        assert_almost_equal(predictions[i], 1.0, atol=1e-5)
        assert_almost_equal(corrected[i], 1.0, atol=1e-5)

    _ = router[0]
    _ = router_norm[0]
    _ = prediction_coefs[0]
    _ = correction_coefs[0]
    _ = output_scale[0]
    _ = streams[0]
    _ = predictions[0]
    _ = corrected[0]
    _ = activated[0]
    _ = scratch[0]


fn main() raises:
    test_forward_altup_predict_correct_identity_contract()
    print("test_altup_contract.mojo passed!")
