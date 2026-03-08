"""Tests for CI and publish workflow contracts."""

from pathlib import Path


def test_ci_workflow_has_gpu_smoke_job() -> None:
    root = Path(__file__).resolve().parents[3]
    ci_path = root / ".github" / "workflows" / "ci.yml"

    workflow_text = ci_path.read_text(encoding="utf-8")

    assert "build-wheels-cuda:" in workflow_text, "CI must include a CUDA wheel build job for smoke testing"
    assert "benchmark_cpu_generation.json" in workflow_text, "CI must save CPU benchmark artifact"


def test_publish_workflow_has_cuda_matrix() -> None:
    root = Path(__file__).resolve().parents[3]
    publish_path = root / ".github" / "workflows" / "publish.yml"

    workflow_text = publish_path.read_text(encoding="utf-8")

    assert "build-wheels-cuda:" in workflow_text, "Publish workflow must include a CUDA wheel matrix"
    assert "upload-artifact" in workflow_text, "Must upload CUDA wheel artifacts"
    assert "benchmark_gpu_generation.json" in workflow_text, "Publish workflow must run and save GPU benchmarks"
