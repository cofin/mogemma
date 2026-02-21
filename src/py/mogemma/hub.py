from pathlib import Path


class HubManager:
    """Manages downloading and caching Gemma 3 models from Hugging Face Hub."""

    def __init__(self, cache_path: str | Path | None = None) -> None:
        """Initialize the HubManager."""
        if cache_path is None:
            self.cache_path = Path.home() / ".cache" / "mogemma"
        else:
            self.cache_path = Path(cache_path)

        self.cache_path.mkdir(parents=True, exist_ok=True)

    def resolve_model(self, model_id: str) -> Path:
        """Resolve a model ID to a local path.
        If model_id is a local path that exists, return that path.
        If model_id exists in this cache, return the cached path.
        Otherwise, return the original model ID to allow hub resolution.
        """
        # 1. Check if model_id is a direct local path
        local_path = Path(model_id)
        if local_path.exists() and local_path.is_dir():
            return local_path

        # 2. Check in our cache
        cached_path = self.cache_path / model_id.replace("/", "--")
        if cached_path.exists() and cached_path.is_dir():
            return cached_path

        # 3. Fallback to HF Hub by preserving the model id string representation.
        return Path(model_id)

    def download(self, model_id: str, **kwargs: object) -> Path:
        """Download a model from the Hugging Face Hub."""
        try:
            from huggingface_hub import snapshot_download
        except ModuleNotFoundError as exc:
            msg = (
                "Hub downloads require optional dependency 'huggingface-hub'. Install with: pip install 'mogemma[hub]'"
            )
            raise ModuleNotFoundError(msg) from exc

        local_dir = self.cache_path / model_id.replace("/", "--")

        path = snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False, **kwargs)
        return Path(path)
