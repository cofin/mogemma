"""Dependency facades used by runtime imports.

This module keeps imports typed consistently and avoids repetitive
`try`/`except ModuleNotFoundError` blocks in feature modules.
"""

from typing import Any

__all__ = ["OPENTELEMETRY_INSTALLED", "TOKENIZERS_INSTALLED", "_TokenizerImpl", "trace"]
_TokenizerImpl: Any | None
trace: Any | None

try:
    from tokenizers import Tokenizer as _TokenizerClass  # type: ignore[import-untyped]
except ModuleNotFoundError:
    _TokenizerImpl = None
else:
    _TokenizerImpl = _TokenizerClass

try:
    from opentelemetry import trace as _trace
except ModuleNotFoundError:
    trace = None
else:
    trace = _trace

TOKENIZERS_INSTALLED = _TokenizerImpl is not None
OPENTELEMETRY_INSTALLED = trace is not None
