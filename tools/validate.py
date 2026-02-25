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
TEXT_MODEL_ID = "gemma3-270m-it"
EMBED_MODEL_ID = "gemma3-270m-it"
NANO_MODEL_ID = "gemma3n-e2b-it"

from mogemma import EmbeddingConfig, EmbeddingModel, GenerationConfig, SyncGemmaModel

_INSTRUCTION_START = "<start_of_turn>"
_INSTRUCTION_END = "<end_of_turn>"
_STANDARD_FACTUAL_MODEL = "gemma3-270m-it"


def _format_instruction_prompt(prompt: str) -> str:
    """Wrap plain prompts in Gemma instruction-turn format."""
    if _INSTRUCTION_START in prompt or _INSTRUCTION_END in prompt:
        return prompt
    return f"{_INSTRUCTION_START}user\n{prompt}\n{_INSTRUCTION_END}\n{_INSTRUCTION_START}model\n"


def _assert_semantic_quality(model_id: str, response: str) -> None:
    """Run deterministic semantic gates for known baseline models."""
    normalized = model_id.removeprefix("google/")
    if normalized != _STANDARD_FACTUAL_MODEL:
        return
    if "paris" not in response.lower():
        msg = (
            f"Semantic validation failed for {model_id}: expected response to mention 'Paris'. "
            f"Received: {response!r}"
        )
        raise ValueError(msg)


def validate_llm_generation(model_id: str):
    print(f"\n[LLM] Validating Generation ({model_id})...")
    # This will trigger an automatic download from GCS if not in cache
    config = GenerationConfig(model_path=model_id, max_tokens=64, temperature=0.0, top_k=1, top_p=1.0)

    try:
        model = SyncGemmaModel(config)
        prompt = "What is the capital of France?"
        prompt_to_send = _format_instruction_prompt(prompt)
        print(f"Prompt: '{prompt}'")
        response = model.generate(prompt_to_send)
        print(f"Response: {response}")
        _assert_semantic_quality(model_id, response)
        print("\nSUCCESS: Text generation works end-to-end.")
    except ValueError as e:
        print(f"\nFAILED: Semantic validation error: {e}")
        sys.exit(1)
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


def validate_embeddings(model_id: str):
    print(f"\n[Embed] Validating Embeddings ({model_id})...")
    config = EmbeddingConfig(model_path=model_id)

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
    parser.add_argument("--model", type=str, help="Model ID or path to use for validation")
    args = parser.parse_args()

    print("--- Starting Mogemma Validation ---")
    print("Models will be downloaded from Google Cloud Storage automatically if missing.")

    # Check if Mojo core exists
    so_path = Path(__file__).parent.parent / "src" / "py" / "mogemma" / "_core.so"
    if not so_path.exists():
        print("WARNING: Mojo shared library (_core.so) not found.")
        print("Run `make build` to compile the bridge before validating.")
        sys.exit(1)

    models_to_test_llm = [args.model] if args.model else [TEXT_MODEL_ID, NANO_MODEL_ID]
    models_to_test_embed = [args.model] if args.model else [EMBED_MODEL_ID, NANO_MODEL_ID]

    if args.mode in ["llm", "both"]:
        for m_id in models_to_test_llm:
            validate_llm_generation(m_id)

    if args.mode in ["embed", "both"]:
        for m_id in models_to_test_embed:
            validate_embeddings(m_id)

    print("\n--- Validation Complete! ---")


if __name__ == "__main__":
    main()
