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

import numpy as np
import pytest
from safetensors.numpy import save_file

from mogemma.config import GenerationConfig
from mogemma.model import SyncGemmaModel

from .parity_config import DETERMINISTIC_PROFILE, PARITY_THRESHOLDS, PERF_THRESHOLDS, PROMPT_FIXTURES


def _create_dummy_safetensors(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create minimal Nano tensors to avoid validation failures in Mojo engine
    hidden_size = 32
    head_dim = 16
    num_heads = 4
    num_kv_heads = 1
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    vocab_size = 128
    intermediate_size = 64
    num_modalities = 4
    per_layer_dim = 16
    
    tensors = {
        "model.embed_tokens.weight": np.zeros((vocab_size, hidden_size), dtype=np.float32),
        "model.norm.weight": np.zeros((hidden_size,), dtype=np.float32),
        "lm_head.weight": np.zeros((vocab_size, hidden_size), dtype=np.float32),
        "model.per_layer_embed.weight": np.zeros((100, 1, per_layer_dim), dtype=np.float32),
        "model.per_layer_embed.projection.weight": np.zeros((hidden_size, 1, per_layer_dim), dtype=np.float32),
        "model.per_layer_embed.norm.weight": np.zeros((per_layer_dim,), dtype=np.float32),
    }
    
    for i in range(3):
        tensors[f"model.altup.projection.{i}.weight"] = np.zeros((hidden_size, hidden_size), dtype=np.float32)
        tensors[f"model.altup.unembed.{i}.weight"] = np.zeros((hidden_size, hidden_size), dtype=np.float32)
        
    pfx = "model.layers.0"
    tensors.update({
        f"{pfx}.input_layernorm.weight": np.zeros((hidden_size,), dtype=np.float32),
        f"{pfx}.post_attention_layernorm.weight": np.zeros((hidden_size,), dtype=np.float32),
        f"{pfx}.self_attn.q_proj.weight": np.zeros((q_size, hidden_size), dtype=np.float32),
        f"{pfx}.self_attn.k_proj.weight": np.zeros((kv_size, hidden_size), dtype=np.float32),
        f"{pfx}.self_attn.v_proj.weight": np.zeros((kv_size, hidden_size), dtype=np.float32),
        f"{pfx}.self_attn.o_proj.weight": np.zeros((hidden_size, q_size), dtype=np.float32),
        f"{pfx}.mlp.gate_proj.weight": np.zeros((intermediate_size, hidden_size), dtype=np.float32),
        f"{pfx}.mlp.up_proj.weight": np.zeros((intermediate_size, hidden_size), dtype=np.float32),
        f"{pfx}.mlp.down_proj.weight": np.zeros((hidden_size, intermediate_size), dtype=np.float32),
        f"{pfx}.self_attn.q_norm.weight": np.zeros((head_dim,), dtype=np.float32),
        f"{pfx}.self_attn.k_norm.weight": np.zeros((head_dim,), dtype=np.float32),
        f"{pfx}.pre_feedforward_layernorm.weight": np.zeros((hidden_size,), dtype=np.float32),
        f"{pfx}.post_feedforward_layernorm.weight": np.zeros((hidden_size,), dtype=np.float32),
        f"{pfx}.altup.router.weight": np.zeros((num_modalities, hidden_size), dtype=np.float32),
        f"{pfx}.altup.router_norm.weight": np.zeros((hidden_size,), dtype=np.float32),
        f"{pfx}.altup.prediction_coefs": np.zeros((num_modalities, num_modalities, num_modalities), dtype=np.float32),
        f"{pfx}.altup.correction_coefs": np.zeros((num_modalities, num_modalities), dtype=np.float32),
        f"{pfx}.altup.output_scale": np.zeros((num_modalities,), dtype=np.float32),
        f"{pfx}.laurel.down_proj.weight": np.zeros((16, hidden_size), dtype=np.float32),
        f"{pfx}.laurel.up_proj.weight": np.zeros((hidden_size, 16), dtype=np.float32),
        f"{pfx}.laurel.norm.weight": np.zeros((hidden_size,), dtype=np.float32),
        f"{pfx}.per_layer_map.gate.weight": np.zeros((per_layer_dim, hidden_size), dtype=np.float32),
        f"{pfx}.per_layer_map.projection.weight": np.zeros((hidden_size, per_layer_dim), dtype=np.float32),
        f"{pfx}.per_layer_map.norm.weight": np.zeros((hidden_size,), dtype=np.float32),
    })
    
    save_file(tensors, model_dir / "model.safetensors")
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
def test_deterministic_token_parity(nano_model_path: Path, mock_tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU and GPU must generate the exact same tokens under deterministic settings."""
    monkeypatch.setenv("MOGEMMA_GPU_AVAILABLE", "1")
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
def test_quality_gate_instruction_prompts(nano_model_path: Path, mock_tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure instruction wrapping and EOS behaviors don't regress to gibberish."""
    monkeypatch.setenv("MOGEMMA_GPU_AVAILABLE", "1")
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
def test_performance_gate_throughput(nano_model_path: Path, mock_tokenizer: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """GPU must meet throughput improvement thresholds compared to CPU baseline."""
    monkeypatch.setenv("MOGEMMA_GPU_AVAILABLE", "1")
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
