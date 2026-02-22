"""Model resolution and Hub download helpers."""

from __future__ import annotations

import re
from pathlib import Path

from .typing import HUGGINGFACE_HUB_INSTALLED, snapshot_download

_HF_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*)/[A-Za-z0-9](?:[A-Za-z0-9._-]*)$")


class HubManager:
    """Manages downloading and caching Gemma 3 models from Hugging Face Hub."""

    def __init__(self, cache_path: str | Path | None = None) -> None:
        """Initialize the HubManager."""
        if cache_path is None:
            self.cache_path = Path.home() / ".cache" / "mogemma"
        else:
            self.cache_path = Path(cache_path)

        self.cache_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _is_hf_model_id(model_id: str) -> bool:
        """Check if a model id looks like a Hugging Face repo id."""
        path = Path(model_id)
        if path.is_absolute():
            return False
        if path.name.startswith("."):
            return False
        if path.exists():
            return False
        if any(part in {".", ".."} for part in path.parts):
            return False
        return _HF_MODEL_ID_RE.match(model_id) is not None

    @staticmethod
    def _cache_dir_for_model_id(cache_root: Path, model_id: str) -> Path:
        return cache_root / model_id.replace("/", "--")

    @staticmethod
    def _download_error(model_id: str, exc: Exception) -> Exception:
        """Format a user-facing exception for failed downloads."""
        detail = str(exc)
        lowered = detail.lower()
        exc_name = type(exc).__name__.lower()

        if "offline" in lowered or "offlinemode" in lowered or isinstance(exc, OSError):
            msg = (
                f"Cannot download '{model_id}' in offline mode or restricted network environments. "
                "Set HF_HUB_OFFLINE=0 and ensure network connectivity before retrying."
            )
            return ConnectionError(msg)

        auth_indicators = ["401", "403", "forbidden", "unauthorized", "gated", "authentication", "token", "oauth"]
        if any(indicator in lowered for indicator in auth_indicators) or "gatedrepoerror" in exc_name:
            msg = (
                f"Cannot download '{model_id}' because access is restricted. "
                "Set HF_TOKEN (or run `huggingface-cli login`) and ensure model access is granted."
            )
            return PermissionError(msg)

        if "not found" in lowered or "404" in lowered or "does not exist" in lowered:
            return FileNotFoundError(f"Model '{model_id}' was not found on Hugging Face Hub: {detail}")

        return RuntimeError(f"Failed to download '{model_id}' from Hugging Face Hub: {detail}")

    def resolve_model(
        self, model_id: str, *, download_if_missing: bool = False, strict: bool = False, **download_kwargs: object
    ) -> Path:
        """Resolve a model ID to a local path.

        If model_id is a local path that exists, return that path.
        If model_id exists in this cache, return the cached path.
        If model_id is an HF id and download_if_missing is False, return the original id.
        If download_if_missing is True, download missing HF ids before returning.
        If strict=True, reject unresolved local paths with a clear contract error.
        """
        # 1. Check if model_id is a direct local path
        local_path = Path(model_id)
        if local_path.exists() and local_path.is_dir():
            return local_path

        if local_path.exists() and not local_path.is_dir():
            msg = f"Model path '{model_id}' exists but is not a directory."
            if strict:
                raise ValueError(msg)
            return local_path

        # 2. Check in our cache
        cached_path = self._cache_dir_for_model_id(self.cache_path, model_id)
        if cached_path.exists() and cached_path.is_dir():
            return cached_path

        # 3. Fallback to HF Hub model id handling.
        if self._is_hf_model_id(model_id) and download_if_missing:
            return self.download(model_id, **download_kwargs)

        if strict:
            msg = (
                f"Cannot resolve model path '{model_id}'. "
                "Use an existing local directory or a valid Hugging Face model id (namespace/model)."
            )
            raise ValueError(msg)

        return Path(model_id)

    def download(self, model_id: str, **kwargs: object) -> Path:
        """Download a model from the Hugging Face Hub."""
        if not HUGGINGFACE_HUB_INSTALLED or snapshot_download is None:
            msg = (
                "Hugging Face Hub dependency is required at install time. "
                "The base package now includes 'huggingface-hub'; reinstall if import fails."
            )
            raise ModuleNotFoundError(msg)

        if "local_dir" in kwargs:
            msg = "'local_dir' is managed by Mogemma cache settings and cannot be overridden."
            raise ValueError(msg)

        local_dir = self._cache_dir_for_model_id(self.cache_path, model_id)

        try:
            path = snapshot_download(repo_id=model_id, local_dir=local_dir, **kwargs)
            return Path(path)
        except Exception as exc:
            raise self._download_error(model_id, exc) from exc
