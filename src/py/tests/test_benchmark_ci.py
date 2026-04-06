"""Tests for the perf-benchmark CI workflow and benchmark script entry point."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "perf-benchmark.yml"


@pytest.fixture
def workflow_text() -> str:
    """Load the perf-benchmark workflow as raw text."""
    assert WORKFLOW_PATH.exists(), f"Workflow not found: {WORKFLOW_PATH}"
    return WORKFLOW_PATH.read_text()


def test_workflow_exists() -> None:
    assert WORKFLOW_PATH.exists(), f"Missing {WORKFLOW_PATH}"


def test_workflow_triggers_on_push_main(workflow_text: str) -> None:
    assert "push:" in workflow_text
    assert "- main" in workflow_text


def test_workflow_triggers_on_pull_request(workflow_text: str) -> None:
    assert "pull_request:" in workflow_text


def test_workflow_runs_make_benchmark(workflow_text: str) -> None:
    assert "make benchmark" in workflow_text


def test_workflow_uploads_artifacts(workflow_text: str) -> None:
    assert "upload-artifact" in workflow_text


def test_benchmark_script_runs_synthetic(tmp_path: Path) -> None:
    """Smoke-test that the benchmark entry point runs with stubs."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "benchmark.py"),
            "--mode",
            "generation",
            "--rounds",
            "2",
            "--warmup",
            "1",
            "--max-new-tokens",
            "8",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"
    payload = json.loads(result.stdout)
    assert payload["mode"] == "generation"
    assert payload["is_synthetic"] is True
