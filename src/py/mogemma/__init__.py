from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .async_model import AsyncGemmaModel
    from .config import EmbeddingConfig, GenerationConfig
    from .hub import HubManager
    from .model import EmbeddingModel, GemmaModel

__all__ = [
    "AsyncGemmaModel",
    "EmbeddingConfig",
    "EmbeddingModel",
    "GemmaModel",
    "GenerationConfig",
    "HubManager",
]

_EXPORT_TO_MODULE = {
    "AsyncGemmaModel": ".async_model",
    "EmbeddingConfig": ".config",
    "EmbeddingModel": ".model",
    "GemmaModel": ".model",
    "GenerationConfig": ".config",
    "HubManager": ".hub",
}

_EXTRA_HINT = {
    "EmbeddingModel": (
        "Install optional runtime deps with: pip install 'mogemma[embed]' "
        "(or 'mogemma[text]' if you need built-in text tokenization)."
    ),
    "GemmaModel": "Install optional runtime deps with: pip install 'mogemma[text]'",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        msg = f"module 'mogemma' has no attribute '{name}'"
        raise AttributeError(msg)

    try:
        module = import_module(module_name, __name__)
    except ModuleNotFoundError as exc:
        hint = _EXTRA_HINT.get(name)
        if hint is None:
            raise
        raise ModuleNotFoundError(f"{name} could not be imported. {hint}") from exc

    value = getattr(module, name)
    globals()[name] = value
    return value
