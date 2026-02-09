from collections.abc import Callable
from contextlib import nullcontext
from functools import wraps
from typing import Any

try:
    from opentelemetry import trace
except ModuleNotFoundError:
    trace = None  # type: ignore[assignment]


class _NoOpTracer:
    def start_as_current_span(self, _: str) -> Any:
        return nullcontext()


tracer = trace.get_tracer("mogemma") if trace is not None else _NoOpTracer()


def trace_inference(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a function with an inference tracing span."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(name):
                return func(*args, **kwargs)

        return wrapper

    return decorator
