import os
import time
from pathlib import Path

from mogemma import GenerationConfig, SyncGemmaModel


def main() -> None:
    """Run the Mojo bridge smoke test."""
    model_name = os.environ.get("MOGEMMA_SMOKE_TEST_MODEL_PATH")
    if not model_name:
        print("Skipping smoke test: set MOGEMMA_SMOKE_TEST_MODEL_PATH to a local model directory.")
        return

    model_path = Path(model_name)
    if not model_path.is_dir():
        print(f"Smoke test failed: '{model_path}' is not an existing directory.")
        raise SystemExit(1)

    config = GenerationConfig(model_path=str(model_path), max_new_tokens=10, temperature=0.7)

    print(f"Initializing SyncGemmaModel with {model_name}...")
    model = SyncGemmaModel(config)

    print("\nGenerating stream:")
    start_time = time.time()
    for token in model.generate_stream("Tell me about Mojo"):
        print(token, end="", flush=True)
        time.sleep(0.1)  # Simulate real-time delay

    end_time = time.time()
    print(f"\n\nStream finished in {end_time - start_time:.2f}s")
    print("Streaming generation simulation successful")


if __name__ == "__main__":
    main()
