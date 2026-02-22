"""Deterministic benchmark harness for release-focused performance baselines."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt

import mogemma.model as model_module
from mogemma import EmbeddingConfig, EmbeddingModel, GenerationConfig, SyncGemmaModel


class _FakeTokenizer:
    """Deterministic tokenizer stub used to avoid external dependencies."""

    @classmethod
    def from_pretrained(cls, _model_path: str) -> "_FakeTokenizer":
        return cls()

    def encode(self, text: str) -> SimpleNamespace:
        del text
        return SimpleNamespace(ids=[11, 22, 33])

    def encode_batch(self, text: list[str] | tuple[str, ...]) -> list[SimpleNamespace]:
        return [SimpleNamespace(ids=[11, 22, 33]) for _ in text]

    def decode(self, tokens: list[int]) -> str:
        del tokens
        return "x"

    def enable_truncation(self, **_: object) -> None:
        return None

    def enable_padding(self, **_: object) -> None:
        return None

    def token_to_id(self, token: str) -> int | None:
        if token in {"</s>", "<eos>", "<|eos|>"}:
            return 0
        return None


class _FakeCore:
    """Core stub with deterministic outputs for benchmark reproducibility."""

    def init_model(self, model_path: str) -> object:
        del model_path
        return object()

    def step(
        self,
        llm: object,
        token_id: int,
        temp: float,
        top_k: int,
        top_p: float,
    ) -> npt.NDArray[np.float32]:
        del llm, temp, top_k, top_p
        base = float(token_id)
        return np.asarray([base + 0.1, base + 0.2, base + 0.3], dtype=np.float32)

    def generate_embeddings(self, llm: object, tokens: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
        del llm
        return np.tile(np.arange(768, dtype=np.float32), (tokens.shape[0], 1))


def _install_stubs() -> None:
    model_module._core = _FakeCore()
    model_module._TokenizerImpl = _FakeTokenizer


def _environment_payload() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
    }


def _run_generation(config: GenerationConfig, prompt: str, *, rounds: int) -> dict[str, object]:
    start = time.perf_counter()
    model = SyncGemmaModel(config)
    outputs: list[str] = []
    for _ in range(rounds):
        outputs.append(model.generate(prompt))
    end = time.perf_counter()
    elapsed = end - start
    tokens = sum(len(text) for text in outputs)
    return {
        "mode": "generation",
        "rounds": rounds,
        "elapsed_s": elapsed,
        "tokens": tokens,
        "tokens_per_second": tokens / elapsed if elapsed > 0 else 0.0,
    }


def _run_embedding(config: EmbeddingConfig, texts: list[str], *, rounds: int) -> dict[str, object]:
    start = time.perf_counter()
    model = EmbeddingModel(config)
    for _ in range(rounds):
        _ = model.embed(texts)
    end = time.perf_counter()
    elapsed = end - start
    return {
        "mode": "embedding",
        "rounds": rounds,
        "elapsed_s": elapsed,
        "input_texts": len(texts),
        "calls_per_second": rounds / elapsed if elapsed > 0 else 0.0,
    }


def _run_benchmark() -> dict[str, object]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["generation", "embedding"],
        default="generation",
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    _install_stubs()
    model_root = Path("benchmark-model")
    model_root.mkdir(exist_ok=True)

    metrics: dict[str, object]
    if args.mode == "generation":
        config = GenerationConfig(model_path=model_root, max_new_tokens=args.max_new_tokens)
        metrics = _run_generation(config, "Benchmark prompt for release parity.", rounds=args.rounds)
    else:
        config = EmbeddingConfig(model_path=model_root)
        metrics = _run_embedding(
            config,
            texts=["Benchmark embedding input one", "Benchmark embedding input two"],
            rounds=args.rounds,
        )
    return {
        "mode": args.mode,
        "model_path": str(model_root),
        "max_new_tokens": args.max_new_tokens,
        "environment": _environment_payload(),
        **metrics,
    }


def main() -> None:
    payload = _run_benchmark()
    print(json.dumps(payload, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
