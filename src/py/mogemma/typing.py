# ruff: noqa: A005

"""Facade re-exports for runtime dependencies."""

from ._typing import OPENTELEMETRY_INSTALLED, SENTENCEPIECE_INSTALLED, _SPProcessorImpl, trace

__all__ = ["OPENTELEMETRY_INSTALLED", "SENTENCEPIECE_INSTALLED", "_SPProcessorImpl", "trace"]
