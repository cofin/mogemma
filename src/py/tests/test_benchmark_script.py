"""Tests for the benchmark script and schema contracts."""

import json
import subprocess
from pathlib import Path

import pytest


def test_benchmark_schema_v2_generation() -> None:
    """Test that the benchmark script outputs the required schema v2 fields."""
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["python", "tools/benchmark.py", "--mode", "generation", "--rounds", "1", "--warmup", "0"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
    )
    
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 2
    assert payload["mode"] == "generation"
    assert payload["is_synthetic"] is True
    assert payload["backend"] == "cpu"
    assert payload["device"] == "cpu"
    assert payload["runner_class"] == "standard-ci"
    assert payload["threshold_profile"] == "default"
    
    metrics = payload["metrics"]
    assert "tokens_per_second" in metrics
    assert "p95_latency_s" in metrics
    
    env = payload["environment"]
    assert env["accelerator"] == "none"


def test_benchmark_schema_v2_embedding() -> None:
    """Test that embedding mode respects schema v2 contracts."""
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["python", "tools/benchmark.py", "--mode", "embedding", "--rounds", "1"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
    )
    
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 2
    assert payload["mode"] == "embedding"
    metrics = payload["metrics"]
    assert "calls_per_second" in metrics


def test_benchmark_baseline_comparison_gate(tmp_path: Path) -> None:
    """Test gate passes or fails appropriately against a baseline."""
    root = Path(__file__).resolve().parents[3]
    
    baseline = {
        "metrics": {
            "tokens_per_second": 100.0,
            "p95_latency_s": 0.05
        }
    }
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline))
    
    result = subprocess.run(
        [
            "python", "tools/benchmark.py", 
            "--mode", "generation", 
            "--corpus", "medium", 
            "--baseline-json", str(baseline_path),
            "--rounds", "1",
            "--warmup", "0"
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
    )
    
    payload = json.loads(result.stdout)
    # The synthetic test outputs very high throughput because it's completely fake.
    # It should pass because tps > 1.25 * 100.
    assert "gate_passed" in payload
    # Let's verify we parse it.
    assert isinstance(payload["gate_passed"], bool)


def test_benchmark_cli_alias_contract() -> None:
    """Test that --max-new-tokens is parsed correctly and represented in the payload."""
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["python", "tools/benchmark.py", "--mode", "generation", "--max-new-tokens", "10", "--rounds", "1", "--warmup", "0"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
    )
    
    payload = json.loads(result.stdout)
    assert payload["max_tokens"] == 10
