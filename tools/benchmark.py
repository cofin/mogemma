import time
from unittest.mock import MagicMock, patch

import numpy as np

from mogemma import GenerationConfig, SyncGemmaModel


def run_benchmark(num_tokens: int = 100) -> float:
    print(f"Benchmarking generation of {num_tokens} tokens...")

    model_name = "bert-base-uncased"
    config = GenerationConfig(model_path=model_name, max_new_tokens=num_tokens)

    with patch("mogemma.model.Tokenizer.from_pretrained") as mock:
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": np.array([[1, 2, 3]], dtype=np.int32)}
        tokenizer.decode.return_value = " "
        mock.return_value = tokenizer

        model = SyncGemmaModel(config)

        start_time = time.perf_counter()
        _ = model.generate("Benchmark prompt")
        end_time = time.perf_counter()

        duration = end_time - start_time
        tps = num_tokens / duration

        print(f"Duration: {duration:.4f}s")
        print(f"Tokens per second (TPS): {tps:.2f}")
        return tps


if __name__ == "__main__":
    run_benchmark()
