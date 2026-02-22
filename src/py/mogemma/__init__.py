from importlib import import_module
from typing import TYPE_CHECKING, Any

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

_EXTRA_HINT = {
    "EmbeddingModel": (
        "Install optional runtime deps with: pip install 'mogemma[embed]' "
        "(or 'mogemma[text]' if you need built-in text tokenization)."
    ),
    "SyncGemmaModel": "Install optional runtime deps with: pip install 'mogemma[text]'",
    "AsyncGemmaModel": "Install optional runtime deps with: pip install 'mogemma[text]'",
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
        msg = f"{name} could not be imported. {hint}"
        raise ModuleNotFoundError(msg) from exc

    value = getattr(module, name)
    globals()[name] = value
    return value
