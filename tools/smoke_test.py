import time

from mogemma import GenerationConfig, SyncGemmaModel


def main() -> None:
    """Run the Mojo bridge smoke test."""
    model_name = "bert-base-uncased"

    config = GenerationConfig(model_path=model_name, max_new_tokens=10, temperature=0.7)

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
