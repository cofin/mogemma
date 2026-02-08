import os
from pathlib import Path
from huggingface_hub import snapshot_download

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
        """
        Resolve a model ID to a local path.
        If model_id is a local path that exists, returns it.
        Otherwise, looks in the cache.
        """
        # 1. Check if model_id is a direct local path
        local_path = Path(model_id)
        if local_path.exists() and local_path.is_dir():
            return local_path

        # 2. Check in our cache
        cached_path = self.cache_path / model_id.replace("/", "--")
        if cached_path.exists():
            return cached_path

        # 3. Fallback to HF Hub (this doesn't download, just returns where it WOULD be)
        return cached_path

    def download(self, model_id: str, **kwargs) -> Path:
        """
        Download a model from the Hugging Face Hub.
        """
        local_dir = self.cache_path / model_id.replace("/", "--")
        
        path = snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            **kwargs
        )
        return Path(path)
