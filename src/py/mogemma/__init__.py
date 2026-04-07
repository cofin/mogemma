"""Public package entrypoint for mogemma lazy exports."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import EmbeddingConfig, GenerationConfig
    from .hub import HubManager
    from .model import AsyncEmbeddingModel, AsyncGemmaModel, SyncEmbeddingModel, SyncGemmaModel

__all__ = [
    "AsyncEmbeddingModel",
    "AsyncGemmaModel",
    "EmbeddingConfig",
    "GenerationConfig",
    "HubManager",
    "SyncEmbeddingModel",
    "SyncGemmaModel",
]

_EXPORT_TO_MODULE = {
    "AsyncEmbeddingModel": ".model",
    "AsyncGemmaModel": ".model",
    "EmbeddingConfig": ".config",
    "GenerationConfig": ".config",
    "HubManager": ".hub",
    "SyncEmbeddingModel": ".model",
    "SyncGemmaModel": ".model",
}

_EXTRA_HINT = {
    "SyncGemmaModel": "Install optional runtime deps with: pip install 'mogemma[llm]'",
    "AsyncGemmaModel": "Install optional runtime deps with: pip install 'mogemma[llm]'",
    "SyncEmbeddingModel": "Install optional runtime deps with: pip install 'mogemma[llm]'",
    "AsyncEmbeddingModel": "Install optional runtime deps with: pip install 'mogemma[llm]'",
}


def __getattr__(name: str) -> object:
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
        msg = f"{name} could not be imported. {hint}"
        raise ModuleNotFoundError(msg) from exc

    value = getattr(module, name)
    globals()[name] = value
    return value
