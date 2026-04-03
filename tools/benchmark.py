"""Deterministic benchmark harness for release-focused performance baselines."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt

import mogemma.model as model_module
from mogemma import EmbeddingConfig, SyncEmbeddingModel, GenerationConfig, SyncGemmaModel


class _FakeTokenizer:
    """Deterministic tokenizer stub used to avoid external dependencies."""

    def __init__(self, model_path: str) -> None:
        del model_path

    @classmethod
    def from_pretrained(cls, _model_path: str) -> _FakeTokenizer:
        return cls(_model_path)

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

    def init_model(self, metadata: dict[str, tuple[int, tuple[int, ...], str]]) -> object:
        del metadata
        return {"pos": 0}

    def step(self, llm: object, token_id: int, temp: float, top_k: int, top_p: float) -> npt.NDArray[np.float32]:
        del llm, temp, top_k, top_p
        base = float(token_id)
        return np.asarray([base + 0.1, base + 0.2, base + 0.3], dtype=np.float32)

    def generate_embeddings(self, llm: object, tokens: list[list[int]]) -> npt.NDArray[np.float32]:
        del llm
        return np.tile(np.arange(768, dtype=np.float32), (len(tokens), 1))


class _FakeLoader:
    """Loader stub that provides empty metadata without requiring real weight files."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)

    def get_tensor_metadata(self) -> dict[str, tuple[int, tuple[int, ...], str]]:
        return {}

    def close(self) -> None:
        pass


def _fake_auto_loader(model_path: str | Path) -> _FakeLoader:
    return _FakeLoader(model_path)


def _install_stubs() -> None:
    model_module._core = _FakeCore()  # noqa: SLF001
    model_module._Tokenizer = _FakeTokenizer  # noqa: SLF001
    model_module.SENTENCEPIECE_INSTALLED = True
    model_module.auto_loader = _fake_auto_loader  # type: ignore[attr-defined]


def _environment_payload(backend: str) -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "accelerator": "cuda" if backend == "gpu" else "none",
    }


def _run_generation(config: GenerationConfig, prompt: str, *, rounds: int, warmup: int = 3) -> dict[str, object]:
    model = SyncGemmaModel(config)
    # Warmup
    for _ in range(warmup):
        model.generate(prompt)

    latencies: list[float] = []
    tokens = 0
    outputs: list[str] = []
    for _ in range(rounds):
        start = time.perf_counter()
        outputs.append(model.generate(prompt))
        end = time.perf_counter()
        latencies.append(end - start)
        tokens += len(outputs[-1])

    elapsed = sum(latencies)
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0

    return {
        "elapsed_s": elapsed,
        "p95_latency_s": p95_latency,
        "tokens": tokens,
        "tokens_per_second": tokens / elapsed if elapsed > 0 else 0.0,
    }


def _run_embedding(config: EmbeddingConfig, texts: list[str], *, rounds: int) -> dict[str, object]:
    start = time.perf_counter()
    model = SyncEmbeddingModel(config)
    for _ in range(rounds):
        _ = model.embed(texts)
    end = time.perf_counter()
    elapsed = end - start
    return {
        "elapsed_s": elapsed,
        "input_texts": len(texts),
        "calls_per_second": rounds / elapsed if elapsed > 0 else 0.0,
    }


_CORPORA = {
    "short": "Summarize the history of AI in one sentence.",
    "medium": (
        "Explain the architecture of a transformer model and how self-attention works. "
        "Provide examples and describe the flow of tensors through the network."
    )
    * 5,
    "long": "Write a detailed novel about a programmer exploring a dystopian future.",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generation", "embedding"], default="generation")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--corpus", choices=["short", "medium", "long"], default="short")
    parser.add_argument("--batch-size", type=int, default=2, help="Number of inputs to pass to the embedding model")
    parser.add_argument("--real-model-path", type=str, default=None)
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--runner-class", type=str, default="standard-ci")
    parser.add_argument("--baseline-json", type=str, default=None)
    return parser.parse_args()


def _run_benchmark() -> dict[str, object]:
    args = _parse_args()

    is_synthetic = not bool(args.real_model_path)
    if args.real_model_path:
        model_root = Path(args.real_model_path)
    else:
        _install_stubs()
        model_root = Path("benchmark-model")
        model_root.mkdir(exist_ok=True)
        (model_root / "tokenizer.model").touch(exist_ok=True)

    metrics: dict[str, object]
    if args.mode == "generation":
        config = GenerationConfig(model_path=model_root, max_tokens=args.max_new_tokens, device=args.device)
        prompt = _CORPORA[args.corpus]
        metrics = _run_generation(config, prompt, rounds=args.rounds, warmup=args.warmup)
    else:
        config = EmbeddingConfig(model_path=model_root, device=args.device)
        texts = [f"Benchmark embedding input {i}" for i in range(args.batch_size)]
        metrics = _run_embedding(config, texts=texts, rounds=args.rounds)

    payload = {
        "schema_version": 2,
        "mode": args.mode,
        "is_synthetic": is_synthetic,
        "model_path": str(model_root),
        "max_tokens": args.max_new_tokens,
        "corpus": args.corpus,
        "backend": args.backend,
        "device": args.device,
        "runner_class": args.runner_class,
        "warmup_rounds": args.warmup,
        "measured_rounds": args.rounds,
        "threshold_profile": "default",
        "environment": _environment_payload(args.backend),
        "metrics": metrics,
    }

    if args.baseline_json:
        baseline_path = Path(args.baseline_json)
        with baseline_path.open("r") as f:
            baseline = json.load(f)

        base_metrics = baseline.get("metrics", {})

        pass_gate = True
        reason = []
        if args.mode == "generation":
            tps = metrics.get("tokens_per_second", 0)
            base_tps = base_metrics.get("tokens_per_second", 0)
            p95 = metrics.get("p95_latency_s", 0)
            base_p95 = base_metrics.get("p95_latency_s", 0)

            if args.corpus in ["medium", "long"] and tps < base_tps * 1.25:
                pass_gate = False
                reason.append(f"Throughput {tps:.2f} is not >= 1.25x baseline {base_tps:.2f}")
            if args.corpus == "short" and p95 > base_p95 * 1.15:
                pass_gate = False
                reason.append(f"p95 latency {p95:.4f} regressed > 15% vs baseline {base_p95:.4f}")
        else:
            cps = metrics.get("calls_per_second", 0)
            base_cps = base_metrics.get("calls_per_second", 0)
            if cps < base_cps * 0.85:
                pass_gate = False
                reason.append(f"Calls per second {cps:.2f} regressed > 15% vs baseline {base_cps:.2f}")

        payload["gate_passed"] = pass_gate
        payload["gate_reason"] = "; ".join(reason) if reason else "Pass"

    return payload


def main() -> None:
    """Run benchmark script."""
    payload = _run_benchmark()
    sys.stdout.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
