"""Tests for CPU vs GPU nano parity gates.

These tests define the quality and correctness gates for GPU nano parity.
They are expected to fail initially as the GPU parity path is incomplete.
"""

import json
import struct
import time
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mogemma.config import GenerationConfig
from mogemma.model import SyncGemmaModel
from parity_config import DETERMINISTIC_PROFILE, PARITY_THRESHOLDS, PERF_THRESHOLDS, PROMPT_FIXTURES


def _create_dummy_safetensors(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "model.safetensors").open("wb") as f:
        h = json.dumps({}).encode("utf-8")
        f.write(struct.pack("<Q", len(h)) + h)
    (model_dir / "tokenizer.model").touch()


@pytest.fixture
def mock_tokenizer() -> Iterator[MagicMock]:
    with patch("mogemma.model._Tokenizer") as mock:
        tokenizer = MagicMock()
        encoded_mock = MagicMock()
        encoded_mock.ids = [1, 2, 3]
        tokenizer.encode.return_value = encoded_mock
        tokenizer.decode.return_value = "dummy"
        tokenizer.token_to_id.return_value = 999
        mock.return_value = tokenizer
        yield tokenizer


@pytest.fixture
def nano_model_path(tmp_path: Path) -> Path:
    # A fake nano model directory
    model_dir = tmp_path / "gemma3n-dummy-it"
    _create_dummy_safetensors(model_dir)
    return model_dir


# 2.1 Failing test harness for CPU-vs-GPU deterministic token comparison
def test_deterministic_token_parity(nano_model_path: Path, mock_tokenizer: MagicMock) -> None:
    """CPU and GPU must generate the exact same tokens under deterministic settings."""
    config_cpu = GenerationConfig(
        model_path=nano_model_path,
        device="cpu",
        temperature=DETERMINISTIC_PROFILE.temperature,
        top_k=DETERMINISTIC_PROFILE.top_k,
        top_p=DETERMINISTIC_PROFILE.top_p,
    )
    config_gpu = GenerationConfig(
        model_path=nano_model_path,
        device="gpu",
        temperature=DETERMINISTIC_PROFILE.temperature,
        top_k=DETERMINISTIC_PROFILE.top_k,
        top_p=DETERMINISTIC_PROFILE.top_p,
    )

    model_cpu = SyncGemmaModel(config_cpu)
    model_gpu = SyncGemmaModel(config_gpu)

    prompt = PROMPT_FIXTURES["medium"]

    # These will fail if the backend is not implemented or parity is missing
    output_cpu = model_cpu.generate(prompt)
    output_gpu = model_gpu.generate(prompt)

    if PARITY_THRESHOLDS.exact_token_parity:
        assert output_cpu == output_gpu, "GPU and CPU generated tokens do not match exactly"


# 2.2 Failing quality-gate tests for instruction-formatted prompts and EOS behavior
def test_quality_gate_instruction_prompts(nano_model_path: Path, mock_tokenizer: MagicMock) -> None:
    """Ensure instruction wrapping and EOS behaviors don't regress to gibberish."""
    config_gpu = GenerationConfig(
        model_path=nano_model_path,
        device="gpu",
        temperature=DETERMINISTIC_PROFILE.temperature,
        top_k=DETERMINISTIC_PROFILE.top_k,
        top_p=DETERMINISTIC_PROFILE.top_p,
    )
    model_gpu = SyncGemmaModel(config_gpu)

    # Test that the prompt encoding respects instruction wrap
    prompt = PROMPT_FIXTURES["gibberish_regression"]
    _ = model_gpu.generate(prompt)

    # Assert tokenizer encode was called with start_of_turn wrapper
    encoded_prompt = mock_tokenizer.encode.call_args.args[0]
    if DETERMINISTIC_PROFILE.instruction_wrap:
        assert "<start_of_turn>user\n" in encoded_prompt


# 2.3 Failing performance gate scaffold with baseline capture and threshold checks
def test_performance_gate_throughput(nano_model_path: Path, mock_tokenizer: MagicMock) -> None:
    """GPU must meet throughput improvement thresholds compared to CPU baseline."""
    config_cpu = GenerationConfig(model_path=nano_model_path, device="cpu", max_tokens=10)
    config_gpu = GenerationConfig(model_path=nano_model_path, device="gpu", max_tokens=10)

    model_cpu = SyncGemmaModel(config_cpu)
    model_gpu = SyncGemmaModel(config_gpu)

    prompt = PROMPT_FIXTURES["long"]

    t0 = time.time()
    _ = model_cpu.generate(prompt)
    cpu_time = time.time() - t0

    t0 = time.time()
    _ = model_gpu.generate(prompt)
    gpu_time = time.time() - t0

    cpu_throughput = 10 / cpu_time if cpu_time > 0 else 0
    gpu_throughput = 10 / gpu_time if gpu_time > 0 else 0

    assert gpu_throughput >= (cpu_throughput * PERF_THRESHOLDS.throughput_improvement_ratio)
