import time
import numpy as np
from pathlib import Path
from mogemma import GenerationConfig, GemmaModel
from unittest.mock import patch, MagicMock

def run_benchmark(num_tokens=100):
    print(f"Benchmarking generation of {num_tokens} tokens...")
    
    model_name = "bert-base-uncased"
    config = GenerationConfig(model_path=model_name, max_new_tokens=num_tokens)
    
    # Mock tokenizer to avoid network
    with patch("mogemma.model.AutoTokenizer.from_pretrained") as mock:
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]], dtype=np.int32)
        }
        tokenizer.decode.return_value = " "
        mock.return_value = tokenizer
        
        model = GemmaModel(config)
        
        start_time = time.perf_counter()
        # Trigger generation
        _ = model.generate("Benchmark prompt")
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        tps = num_tokens / duration
        
        print(f"Duration: {duration:.4f}s")
        print(f"Tokens per second (TPS): {tps:.2f}")
        return tps

if __name__ == "__main__":
    run_benchmark()
