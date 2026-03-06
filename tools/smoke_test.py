"""Smoke test script for Mojo bridge initialization.

Verifies basic runtime configuration and text generation capabilities
without invoking full correctness suites.
"""

import os
import sys
import time
from pathlib import Path

from mogemma import GenerationConfig, SyncGemmaModel


def main() -> None:
    """Run the Mojo bridge smoke test."""
    model_name = os.environ.get("MOGEMMA_SMOKE_TEST_MODEL", "gemma3n-e2b-it")
    # This automatically downloads the model using the HubManager if missing
    config = GenerationConfig(model_path=model_name, max_tokens=15, temperature=0.0, top_k=1, top_p=1.0)

    so_path = Path(__file__).parent.parent / "src" / "py" / "mogemma" / "_core.so"
    if not so_path.exists():
        sys.stdout.write("Mojo bridge not found. Run 'make build' first.\n")
        sys.exit(1)

    sys.stdout.write(f"Initializing Mojo engine with {model_name}...\n")
    try:
        model = SyncGemmaModel(config)
    except RuntimeError as e:
        sys.stdout.write(f"Failed to initialize model: {e}\n")
        sys.exit(1)

    sys.stdout.write("Engine initialized. Running generation...\n")

    start_time = time.time()
    prompt = "<start_of_turn>user\nTest prompt<end_of_turn>\n<start_of_turn>model\n"
    response = model.generate(prompt)

    sys.stdout.write("\n=== Response ===\n")
    sys.stdout.write(f"{response}\n")
    sys.stdout.write("================\n")

    sys.stdout.write("\nSimulating stream...\n")
    for word in response.split():
        sys.stdout.write(f"{word} ")
        sys.stdout.flush()
        time.sleep(0.1)  # Simulate real-time delay

    end_time = time.time()
    sys.stdout.write(f"\n\nStream finished in {end_time - start_time:.2f}s\n")
    sys.stdout.write("Streaming generation simulation successful\n")


if __name__ == "__main__":
    main()
