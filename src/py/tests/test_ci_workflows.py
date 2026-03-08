"""Tests for CI and publish workflow contracts."""

from pathlib import Path


def test_ci_workflow_has_cpu_benchmark() -> None:
    root = Path(__file__).resolve().parents[3]
    ci_path = root / ".github" / "workflows" / "ci.yml"

    workflow_text = ci_path.read_text(encoding="utf-8")

    assert "benchmark_cpu_generation.json" in workflow_text, "CI must save CPU benchmark artifact"
