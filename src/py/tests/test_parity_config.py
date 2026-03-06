"""Tests for nano parity gate configurations."""

from parity_config import (
    DETERMINISTIC_PROFILE,
    PROMPT_FIXTURES,
    PARITY_THRESHOLDS,
    PERF_THRESHOLDS
)

def test_deterministic_profile() -> None:
    """Test deterministic decode profile constants."""
    assert DETERMINISTIC_PROFILE.temperature == 0.0
    assert DETERMINISTIC_PROFILE.top_k == 1
    assert DETERMINISTIC_PROFILE.top_p == 1.0

def test_prompt_fixtures() -> None:
    """Test prompt fixtures matrix."""
    assert "short" in PROMPT_FIXTURES
    assert "eos_sensitive" in PROMPT_FIXTURES
    assert "kv_share_boundary" in PROMPT_FIXTURES
    assert "gibberish_regression" in PROMPT_FIXTURES

def test_thresholds() -> None:
    """Test pass/fail thresholds."""
    assert PARITY_THRESHOLDS.logits_atol <= 1e-4
    assert PERF_THRESHOLDS.throughput_improvement_ratio >= 1.25
