import importlib.util
from pathlib import Path

import pytest

_VALIDATE_PATH = Path(__file__).resolve().parents[3] / "tools" / "validate.py"
_SPEC = importlib.util.spec_from_file_location("validate_module", _VALIDATE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_assert_semantic_quality = _MODULE._assert_semantic_quality
_format_instruction_prompt = _MODULE._format_instruction_prompt


def test_format_instruction_prompt_wraps_plain_prompt() -> None:
    prompt = _format_instruction_prompt("What is the capital of France?")
    assert prompt.startswith("<start_of_turn>user\n")
    assert prompt.endswith("<start_of_turn>model\n")


def test_format_instruction_prompt_keeps_existing_template() -> None:
    prompt = "<start_of_turn>user\nhi\n<end_of_turn>\n<start_of_turn>model\n"
    assert _format_instruction_prompt(prompt) == prompt


def test_assert_semantic_quality_requires_paris_for_standard_model() -> None:
    with pytest.raises(ValueError, match="Paris"):
        _assert_semantic_quality("gemma3-270m-it", "The capital of France is Rome.")


def test_assert_semantic_quality_skips_non_standard_model() -> None:
    _assert_semantic_quality("gemma3n-e2b-it", "incoherent output is not gated here")

def test_validate_llm_generation_exits_on_gpu_unavailable(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import sys
    from unittest.mock import MagicMock
    
    mock_model = MagicMock()
    mock_model.side_effect = RuntimeError("Requested device 'gpu' is unavailable on this host.")
    monkeypatch.setattr(_MODULE, "SyncGemmaModel", mock_model)
    
    with pytest.raises(SystemExit) as exc_info:
        _MODULE.validate_llm_generation("gemma3-270m-it", device="gpu")
        
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Requested device 'gpu' is unavailable" in captured.out
