from testing import assert_equal
from inference import InferenceEngine
from cache import KVCache
from vision import VisionProcessor
from python import Python

def test_multimodal_step():
    var cache = KVCache(num_layers=1, batch_size=1, max_seq_len=10, head_dim=4, num_heads=1)
    var engine = InferenceEngine(cache)
    var processor = VisionProcessor()
    
    var np = Python.import_module("numpy")
    var shape = Python.list()
    shape.append(224)
    shape.append(224)
    shape.append(3)
    var image = np.zeros(shape, dtype=np.uint8)
    
    var visual_embedding = processor.preprocess(image)
    
    var logits = engine.step_vision(visual_embedding)
    assert_equal(len(logits), 256000)

def main():
    try:
        test_multimodal_step()
        print("Test Passed")
    except e:
        print("Test Failed:", e)
