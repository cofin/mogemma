"""Script to validate real end-to-end inference for text and embeddings.

This script requires:
1. The Mojo bridge built (run `make build`).
2. An active internet connection to download models from Google Cloud Storage on first run.
"""

import argparse
import sys
from pathlib import Path

# Ensure we can import mogemma
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "py"))

# Default models for validation
# Note: Gemma 3 Nano (gemma3n-*) uses a different architecture (AltUp, Laurel)
# that the Mojo backend does not yet support. Use standard Gemma 3 models.
TEXT_MODEL_ID = "gemma3-270m-it"
EMBED_MODEL_ID = "gemma3-270m-it"

from mogemma import EmbeddingConfig, EmbeddingModel, GenerationConfig, SyncGemmaModel


def validate_llm_generation():
    print(f"\n[LLM] Validating Generation ({TEXT_MODEL_ID})...")
    # This will trigger an automatic download from GCS if not in cache
    config = GenerationConfig(model_path=TEXT_MODEL_ID, max_new_tokens=32, temperature=0.7)

    try:
        model = SyncGemmaModel(config)
        prompt = "What is the capital of France?"
        print(f"Prompt: '{prompt}'")
        print("Response: ", end="", flush=True)
        for token in model.generate_stream(prompt):
            print(token, end="", flush=True)
        print("\nSUCCESS: Text generation works end-to-end.")
    except RuntimeError as e:
        if "Mojo core is unavailable" in str(e):
            print("\nERROR: Mojo bridge not built. Run `make build` first.")
        elif "No module named 'max'" in str(e):
            print("\nERROR: Modular MAX Engine not found.")
            print("Try: pip install modular --index https://whl.modular.com/nightly/simple/ --prerelease allow")
        else:
            print(f"\nERROR during text generation: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nFAILED: Unexpected text generation error: {e}")
        sys.exit(1)


def validate_embeddings():
    print(f"\n[Embed] Validating Embeddings ({EMBED_MODEL_ID})...")
    config = EmbeddingConfig(model_path=EMBED_MODEL_ID)

    try:
        model = EmbeddingModel(config)
        texts = ["The quick brown fox jumps over the lazy dog.", "MAX Engine is fast."]
        print(f"Input: {texts}")
        embeddings = model.embed(texts)
        print(f"SUCCESS: Generated embeddings with shape {embeddings.shape} (DType: {embeddings.dtype})")
    except Exception as e:
        print(f"\nFAILED: Embedding error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Mogemma End-to-End Validator")
    parser.add_argument("--mode", choices=["llm", "embed", "both"], default="both", help="Validation mode")
    args = parser.parse_args()

    print("--- Starting Mogemma Validation ---")
    print("Models will be downloaded from Google Cloud Storage automatically if missing.")

    # Check if Mojo core exists
    so_path = Path(__file__).parent.parent / "src" / "py" / "mogemma" / "_core.so"
    if not so_path.exists():
        print("WARNING: Mojo shared library (_core.so) not found.")
        print("Run `make build` to compile the bridge before validating.")
        sys.exit(1)

    if args.mode in ["llm", "both"]:
        validate_llm_generation()

    if args.mode in ["embed", "both"]:
        validate_embeddings()

    print("\n--- Validation Complete! ---")


if __name__ == "__main__":
    main()
