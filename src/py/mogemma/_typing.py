"""Dependency facades used by runtime imports.

This module keeps imports typed consistently and avoids repetitive
`try`/`except ModuleNotFoundError` blocks in feature modules.
"""

from typing import Any

__all__ = [
    "OPENTELEMETRY_INSTALLED",
    "SENTENCEPIECE_INSTALLED",
    "TOKENIZERS_INSTALLED",
    "_SPProcessorImpl",
    "_TokenizerImpl",
    "trace",
]
_SPProcessorImpl: Any | None
_TokenizerImpl: Any | None = None
trace: Any | None

try:
    import sentencepiece as _sp  # type: ignore[import-untyped]

    _SPProcessorImpl = _sp.SentencePieceProcessor
except ModuleNotFoundError:
    _SPProcessorImpl = None

try:
    from opentelemetry import trace as _trace
except ModuleNotFoundError:
    trace = None
else:
    trace = _trace

SENTENCEPIECE_INSTALLED = _SPProcessorImpl is not None
TOKENIZERS_INSTALLED = False
OPENTELEMETRY_INSTALLED = trace is not None
