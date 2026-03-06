"""Tests to ensure documentation contracts are maintained."""

from pathlib import Path
import re


def test_release_runbook_has_gpu_sections() -> None:
    root = Path(__file__).resolve().parents[3]
    doc_path = root / "docs" / "release-runbook.md"
    
    content = doc_path.read_text(encoding="utf-8")
    
    assert "## GPU Preflight and Deployment Checks" in content
    assert "## Fallback and Rollback Procedures" in content
    assert "## Incident Triage" in content
    assert "## Owner/Escalation Tables" in content


def test_performance_baselines_has_backend_normalization() -> None:
    root = Path(__file__).resolve().parents[3]
    doc_path = root / "docs" / "performance-baselines.md"
    
    content = doc_path.read_text(encoding="utf-8")
    
    assert "## Runner Normalization" in content
    assert "## Regression Interpretation and Acceptance Thresholds" in content
    assert "`accelerator` value" in content
