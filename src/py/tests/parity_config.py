"""Configuration and constants for parity gates.

Defines the deterministic decode profile, prompt fixtures, and pass/fail
thresholds for GPU-vs-CPU parity tests. Consumed by ``tools/run_parity_gates.py``.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DeterministicDecodeProfile:
    """Profile for deterministic decoding to ensure parity."""

    temperature: float = 0.0
    top_k: int = 1
    top_p: float = 1.0
    seed: int = 42
    instruction_wrap: bool = True
    eos_policy: str = "end_of_turn"


DETERMINISTIC_PROFILE = DeterministicDecodeProfile()


PROMPT_FIXTURES: dict[str, str] = {
    "short": "What is 2+2?",
    "medium": "Write a python function to compute the fibonacci sequence.",
    "long": "Explain the architectural differences between a standard Transformer and the Gemma 4 architecture.",
    "eos_sensitive": "Respond exactly with 'OK' and nothing else.",
    "kv_share_boundary": "A " * 256,  # 256 tokens to cross a potential KV share boundary
    "gibberish_regression": "The quick brown fox jumps over the lazy dog.",
}


@dataclass(frozen=True)
class ParityThresholds:
    """Thresholds for logits and token parity."""

    logits_atol: float = 1e-4
    logits_rtol: float = 1e-4
    exact_token_parity: bool = True


PARITY_THRESHOLDS = ParityThresholds()


@dataclass(frozen=True)
class PerfThresholds:
    """Thresholds for performance parity (release and default-enable)."""

    throughput_improvement_ratio: float = 1.25  # GPU throughput should be >= 1.25x CPU throughput
    latency_p95_regression_ratio: float = 1.15  # GPU latency p95 should be <= 1.15x CPU latency


PERF_THRESHOLDS = PerfThresholds()
