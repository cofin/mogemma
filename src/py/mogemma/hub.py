"""Model resolution and Hub download helpers."""

from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path

import obstore as obs

logger = logging.getLogger(__name__)


class HubManager:
    """Manages downloading and caching Gemma 3 models directly from Google Cloud Storage."""

    def __init__(self, cache_path: str | Path | None = None) -> None:
        """Initialize the HubManager."""
        if cache_path is None:
            self.cache_path = Path.home() / ".cache" / "mogemma"
        else:
            self.cache_path = Path(cache_path)

        self.cache_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _clean_model_id(model_id: str) -> str:
        """Normalize model id to match GCS bucket structures."""
        clean_id = model_id.removeprefix("google/")
        return clean_id.replace("gemma-", "gemma") if clean_id.startswith("gemma-") else clean_id

    @staticmethod
    def _cache_dir_for_model_id(cache_root: Path, model_id: str) -> Path:
        return cache_root / model_id.replace("/", "--")

    def resolve_model(
        self, model_id: str, *, download_if_missing: bool = False, strict: bool = False, **_kwargs: object
    ) -> Path:
        """Resolve a model ID to a local path."""
        local_path = Path(model_id)
        if local_path.exists() and local_path.is_dir():
            return local_path

        if local_path.exists() and not local_path.is_dir():
            msg = f"Model path '{model_id}' exists but is not a directory."
            if strict:
                raise ValueError(msg)
            return local_path

        cached_path = self._cache_dir_for_model_id(self.cache_path, model_id)
        if cached_path.exists() and cached_path.is_dir() and any(cached_path.iterdir()):
            return cached_path

        if download_if_missing:
            return self.download(model_id)

        if strict:
            msg = (
                f"Cannot resolve model path '{model_id}'. "
                "Use an existing local directory or a valid Google model id (e.g., gemma-3-1b-it)."
            )
            raise ValueError(msg)

        return Path(model_id)

    class GCSDownloadError(ConnectionError):
        """Raised when a GCS download fails."""

    class ModelNotFoundError(FileNotFoundError):
        """Raised when a model is not found in the public bucket."""

    def _get_tokenizer_path(self, clean_id: str) -> str | None:
        """Determine the tokenizer path based on model family."""
        if "gemma3n" in clean_id:
            return "tokenizers/tokenizer_gemma3n.model"
        if "gemma3" in clean_id:
            return "tokenizers/tokenizer_gemma3.model"
        if "gemma2" in clean_id:
            return "tokenizers/tokenizer_gemma2.model"
        return None

    def _list_remote_files(self, store: obs.store.GCSStore, prefix: str, clean_id: str) -> list[str]:
        """List all files under the given prefix in GCS."""
        paths = []
        try:
            for page in obs.list(store, prefix):
                items = page if isinstance(page, list) else [page]
                for item in items:
                    path = item["path"]  # pyright: ignore[reportCallIssue,reportArgumentType]
                    if not path.endswith("_$folder$"):
                        paths.append(path)
        except Exception as exc:
            msg = f"Failed to list model {clean_id} from GCS: {exc}"
            raise self.GCSDownloadError(msg) from exc
        return paths

    def download(self, model_id: str) -> Path:
        """Download a model directly from Google Cloud Storage using obstore."""
        clean_id = self._clean_model_id(model_id)
        local_dir = self._cache_dir_for_model_id(self.cache_path, model_id)
        local_dir.mkdir(parents=True, exist_ok=True)

        store = obs.store.GCSStore("gemma-data", config={"skip_signature": "true"})  # type: ignore[arg-type]
        prefix = f"checkpoints/{clean_id}/"

        paths_to_download = self._list_remote_files(store, prefix, clean_id)
        if not paths_to_download:
            msg = f"Model '{clean_id}' was not found in the public gemma-data bucket."
            raise self.ModelNotFoundError(msg)

        tokenizer_path = self._get_tokenizer_path(clean_id)
        if tokenizer_path:
            paths_to_download.append(tokenizer_path)

        def _download_file(remote_path: str) -> None:
            data = obs.get(store, remote_path)
            rel_path = "tokenizer.model" if remote_path == tokenizer_path else remote_path.removeprefix(prefix)
            out_file = local_dir / rel_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with out_file.open("wb") as f:
                f.writelines(data.stream())

        logger.info("Downloading %d files for %s from Google Cloud Storage...", len(paths_to_download), model_id)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(_download_file, paths_to_download))

        return local_dir
