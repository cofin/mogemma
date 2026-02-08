from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

# Initialize global tracer
_provider = TracerProvider()
_processor = SimpleSpanProcessor(ConsoleSpanExporter())
_provider.add_span_processor(_processor)
trace.set_tracer_provider(_provider)

tracer = trace.get_tracer("mogemma")

def trace_inference(name: str):
    """Decorator for tracing inference steps."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(name) as span:
                result = func(*args, **kwargs)
                return result
        return wrapper
    return decorator
