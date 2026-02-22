"""Optional dependency facades used by runtime imports.

This module keeps optional imports typed consistently and avoids repetitive
`try`/`except ModuleNotFoundError` blocks in feature modules.
"""

from collections.abc import Callable
from typing import Any, cast

__all__ = [
    "HUGGINGFACE_HUB_INSTALLED",
    "OPENTELEMETRY_INSTALLED",
    "TOKENIZERS_INSTALLED",
    "_TokenizerImpl",
    "snapshot_download",
    "trace",
]
_TokenizerImpl: Any | None
trace: Any | None

try:
    from huggingface_hub import snapshot_download as _snapshot_download
except ModuleNotFoundError:
    snapshot_download: Callable[..., str] | None = None
else:
    snapshot_download = cast("Callable[..., str]", _snapshot_download)

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

HUGGINGFACE_HUB_INSTALLED = snapshot_download is not None
TOKENIZERS_INSTALLED = _TokenizerImpl is not None
OPENTELEMETRY_INSTALLED = trace is not None
