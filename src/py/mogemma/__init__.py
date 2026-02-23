"""Public package entrypoint for mogemma lazy exports."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import EmbeddingConfig, GenerationConfig
    from .hub import HubManager
    from .model import AsyncGemmaModel, EmbeddingModel, SyncGemmaModel

__all__ = ["AsyncGemmaModel", "EmbeddingConfig", "EmbeddingModel", "GenerationConfig", "HubManager", "SyncGemmaModel"]

_EXPORT_TO_MODULE = {
    "AsyncGemmaModel": ".model",
    "EmbeddingConfig": ".config",
    "EmbeddingModel": ".model",
    "GenerationConfig": ".config",
    "HubManager": ".hub",
    "SyncGemmaModel": ".model",
}


def __getattr__(name: str) -> object:
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        msg = f"module 'mogemma' has no attribute '{name}'"
        raise AttributeError(msg)

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
