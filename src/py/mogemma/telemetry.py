from contextlib import nullcontext
from typing import Any

try:
    from opentelemetry import trace
except ModuleNotFoundError:
    trace = None  # type: ignore[assignment]


class _NoOpTracer:
    def start_as_current_span(self, _: str) -> Any:
        return nullcontext()


tracer = trace.get_tracer("mogemma") if trace is not None else _NoOpTracer()
