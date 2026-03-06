"""Script to validate real end-to-end inference for text and embeddings.

This script requires:
1. The Mojo bridge built (run `make build`).
2. An active internet connection to download models from Google Cloud Storage on first run.
"""

import argparse
import sys
from pathlib import Path

# Ensure we import the local source tree instead of any installed wheel
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "py"))

from mogemma import EmbeddingConfig, EmbeddingModel, GenerationConfig, SyncGemmaModel

TEXT_MODEL_ID = "gemma3-270m-it"
EMBED_MODEL_ID = "gemma3-270m-it"
NANO_MODEL_ID = "gemma3n-e2b-it"


_INSTRUCTION_START = "<start_of_turn>"
_INSTRUCTION_END = "<end_of_turn>\n"
def _format_instruction_prompt(user_text: str) -> str:
    if _INSTRUCTION_START in user_text:
        return user_text
    return f"{_INSTRUCTION_START}user\n{user_text}{_INSTRUCTION_END}{_INSTRUCTION_START}model\n"


def _assert_semantic_quality(model_id: str, response: str) -> None:
    # A lightweight quality gate to catch egregious failures (gibberish, endless padding)
    if "gemma3n" in model_id:
        return
    clean_resp = response.strip().lower()
    min_len = 5
    if len(clean_resp) < min_len or "paris" not in clean_resp:
        msg = (
            f"Semantic validation failed. Output does not look like a valid answer. "
            f"Received: {response!r}"
        )
        raise ValueError(msg)

def validate_llm_generation(model_id: str, device: str = "cpu") -> None:
    """Validate text generation logic for a specific model ID."""
    sys.stdout.write(f"\n[LLM] Validating Generation ({model_id}) on device '{device}'...\n")
    # This will trigger an automatic download from GCS if not in cache
    config = GenerationConfig(model_path=model_id, device=device, max_tokens=64, temperature=0.0, top_k=1, top_p=1.0)

    try:
        model = SyncGemmaModel(config)
        prompt = "What is the capital of France?"
        prompt_to_send = _format_instruction_prompt(prompt)
        sys.stdout.write(f"Prompt: '{prompt}'\n")
        response = model.generate(prompt_to_send)
        sys.stdout.write(f"Response: {response}\n")
        _assert_semantic_quality(model_id, response)
        sys.stdout.write("\nSUCCESS: Text generation works end-to-end.\n")
    except ValueError as e:
        sys.stdout.write(f"\nFAILED: Semantic validation error: {e}\n")
        sys.exit(1)
    except RuntimeError as e:
        if "Mojo core is unavailable" in str(e):
            sys.stdout.write("\nERROR: Mojo bridge not built. Run `make build` first.\n")
        elif "No module named 'max'" in str(e):
            sys.stdout.write("\nERROR: Modular MAX Engine not found.\n")
            sys.stdout.write(
                "Try: pip install modular --index https://whl.modular.com/nightly/simple/ --prerelease allow\n"
            )
        else:
            sys.stdout.write(f"\nERROR during text generation: {e}\n")
        sys.exit(1)


def validate_embeddings(model_id: str, device: str = "cpu") -> None:
    """Validate embedding generation logic for a specific model ID."""
    sys.stdout.write(f"\n[Embed] Validating Embeddings ({model_id}) on device '{device}'...\n")
    config = EmbeddingConfig(model_path=model_id, device=device)

    try:
        model = EmbeddingModel(config)
        texts = ["The quick brown fox jumps over the lazy dog.", "MAX Engine is fast."]
        sys.stdout.write(f"Input: {texts}\n")
        embeddings = model.embed(texts)
        sys.stdout.write(f"SUCCESS: Generated embeddings with shape {embeddings.shape} (DType: {embeddings.dtype})\n")
    except RuntimeError as e:
        sys.stdout.write(f"\nFAILED: Embedding error: {e}\n")
        sys.exit(1)


def main() -> None:
    """Run validation checks across different models and modalities."""
    parser = argparse.ArgumentParser(description="Mogemma End-to-End Validator")
    parser.add_argument("--mode", choices=["llm", "embed", "both"], default="both", help="Validation mode")
    parser.add_argument("--model", type=str, help="Model ID or path to use for validation")
    parser.add_argument("--device", type=str, default="cpu", help="Device to validate (cpu, gpu)")
    args = parser.parse_args()

    sys.stdout.write("--- Starting Mogemma Validation ---\n")
    sys.stdout.write("Models will be downloaded from Google Cloud Storage automatically if missing.\n")

    # Check if Mojo core exists
    so_path = Path(__file__).parent.parent / "src" / "py" / "mogemma" / "_core.so"
    if not so_path.exists():
        sys.stdout.write("WARNING: Mojo shared library (_core.so) not found.\n")
        sys.stdout.write("Run `make build` to compile the bridge before validating.\n")
        sys.exit(1)

    models_to_test_llm = [args.model] if args.model else [TEXT_MODEL_ID, NANO_MODEL_ID]
    models_to_test_embed = [args.model] if args.model else [EMBED_MODEL_ID, NANO_MODEL_ID]

    if args.mode in ["llm", "both"]:
        for m_id in models_to_test_llm:
            validate_llm_generation(m_id, args.device)

    if args.mode in ["embed", "both"]:
        for m_id in models_to_test_embed:
            validate_embeddings(m_id, args.device)

    sys.stdout.write("\n--- Validation Complete! ---\n")


if __name__ == "__main__":
    main()
