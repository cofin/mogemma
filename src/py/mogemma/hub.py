"""Model resolution and Hub download helpers."""

from pathlib import Path

from .typing import snapshot_download


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
        return "/" in model_id and not path.exists()

    def resolve_model(self, model_id: str, *, download_if_missing: bool = False, **download_kwargs: object) -> Path:
        """Resolve a model ID to a local path.

        If model_id is a local path that exists, return that path.
        If model_id exists in this cache, return the cached path.
        If model_id is an HF id and download_if_missing is False, return the original id.
        If download_if_missing is True, download missing HF ids before returning.
        """
        # 1. Check if model_id is a direct local path
        local_path = Path(model_id)
        if local_path.exists() and local_path.is_dir():
            return local_path

        # 2. Check in our cache
        cached_path = self.cache_path / model_id.replace("/", "--")
        if cached_path.exists() and cached_path.is_dir():
            return cached_path

        # 3. Fallback to HF Hub model id handling.
        if self._is_hf_model_id(model_id) and download_if_missing:
            return self.download(model_id, **download_kwargs)
        return Path(model_id)

    def download(self, model_id: str, **kwargs: object) -> Path:
        """Download a model from the Hugging Face Hub."""
        if snapshot_download is None:
            msg = (
                "Hub downloads require optional dependency 'huggingface-hub'. Install with: pip install 'mogemma[hub]'"
            )
            raise ModuleNotFoundError(msg)

        local_dir = self.cache_path / model_id.replace("/", "--")

        path = snapshot_download(repo_id=model_id, local_dir=local_dir, **kwargs)
        return Path(path)
