from testing import assert_equal
from cache import KVCache
# from inference import InferenceLoop # Will create this

from inference import InferenceEngine

def test_inference_step():
    var cache = KVCache(num_layers=1, batch_size=1, max_seq_len=10, head_dim=4, num_heads=1)
    var engine = InferenceEngine(cache)
    var logits = engine.step(token_id=1)
    assert_equal(len(logits), 256000) # Gemma 3 vocab size

def main():
    try:
        test_inference_step()
        print("Test Passed")
    except e:
        print("Test Failed:", e)
