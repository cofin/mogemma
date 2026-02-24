import pytest
import importlib.util
from pathlib import Path


_VALIDATE_PATH = Path(__file__).resolve().parents[3] / "tools" / "validate.py"
_SPEC = importlib.util.spec_from_file_location("validate_module", _VALIDATE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
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
