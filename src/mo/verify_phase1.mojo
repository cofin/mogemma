from testing import assert_equal
from cache import KVCache
from inference import InferenceEngine
from sampling import Sampler
from python import Python

def verify_full_flow():
    print("Starting full flow verification...")
    
    # 1. Initialize Cache
    var cache = KVCache(num_layers=2, batch_size=1, max_seq_len=128, head_dim=64, num_heads=8)
    
    # 2. Initialize Engine
    var engine = InferenceEngine(cache)
    
    # 3. Initialize Sampler
    var sampler = Sampler()
    
    # 4. Simulate Generation Loop
    var np = Python.import_module("numpy")
    var l = Python.list()
    l.append(1)
    l.append(2)
    l.append(3)
    var input_tokens = np.array(l, dtype=np.int32)
    
    print("Running 5-step generation simulation...")
    for i in range(5):
        var logits = engine.step(10 + i) # Dummy tokens
        var next_token = sampler.greedy(logits)
        assert_equal(next_token, 0) # Our dummy engine returns zeros
        print("  Step", i, "token:", next_token)
        
    print("Generation simulation successful")

def main():
    try:
        verify_full_flow()
    except e:
        print("Verification Failed:", e)