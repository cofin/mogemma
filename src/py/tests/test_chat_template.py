"""Tests for Gemma 4 chat template formatting."""

from __future__ import annotations

from mogemma.model import _format_gemma4_prompt


class TestPlainStringPrompt:
    def test_wraps_in_user_turn(self) -> None:
        result = _format_gemma4_prompt("Hello")
        assert result == "<start_of_turn>user\nHello\n<end_of_turn>\n<start_of_turn>model\n"

    def test_passthrough_if_already_formatted(self) -> None:
        pre_formatted = "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n"
        result = _format_gemma4_prompt(pre_formatted)
        assert result == pre_formatted


class TestSystemPrompt:
    def test_system_plus_user(self) -> None:
        result = _format_gemma4_prompt("What is 2+2?", system_prompt="You are a math tutor.")
        expected = (
            "<start_of_turn>system\n"
            "You are a math tutor.\n"
            "<end_of_turn>\n"
            "<start_of_turn>user\n"
            "What is 2+2?\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        assert result == expected

    def test_system_prompt_none_is_ignored(self) -> None:
        result = _format_gemma4_prompt("Hello", system_prompt=None)
        assert result == "<start_of_turn>user\nHello\n<end_of_turn>\n<start_of_turn>model\n"


class TestMultiTurnMessages:
    def test_simple_multi_turn(self) -> None:
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "model", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = _format_gemma4_prompt(messages)
        expected = (
            "<start_of_turn>user\nHi\n<end_of_turn>\n"
            "<start_of_turn>model\nHello!\n<end_of_turn>\n"
            "<start_of_turn>user\nHow are you?\n<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        assert result == expected

    def test_multi_turn_with_system(self) -> None:
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ]
        result = _format_gemma4_prompt(messages)
        expected = (
            "<start_of_turn>system\nBe concise.\n<end_of_turn>\n"
            "<start_of_turn>user\nHi\n<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        assert result == expected

    def test_multi_turn_system_prompt_kwarg_ignored_when_messages_have_system(self) -> None:
        """When messages include a system role, the system_prompt kwarg is ignored."""
        messages = [
            {"role": "system", "content": "From messages."},
            {"role": "user", "content": "Hi"},
        ]
        result = _format_gemma4_prompt(messages, system_prompt="From kwarg.")
        assert "<start_of_turn>system\nFrom messages.\n<end_of_turn>" in result
        assert "From kwarg." not in result
