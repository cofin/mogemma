"""Tracing helpers for optional OpenTelemetry spans."""

from contextlib import AbstractContextManager

from .typing import trace


class _NoOpSpan:
    def set_attribute(self, _key: str, _value: object) -> None:
        return None


class _NoOpSpanManager(AbstractContextManager[_NoOpSpan]):
    def __enter__(self) -> _NoOpSpan:
        return _NoOpSpan()

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object) -> None:
        return None


class _NoOpTracer:
    def start_as_current_span(self, _: str) -> AbstractContextManager[_NoOpSpan]:
        return _NoOpSpanManager()


tracer = trace.get_tracer("mogemma") if trace is not None else _NoOpTracer()
