"""Model resolution and Hub download helpers."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
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

    @staticmethod
    def _has_safetensors(path: Path) -> bool:
        """Return ``True`` when *path* contains ready-to-use safetensors files."""
        return (path / "model.safetensors").exists() or (path / "model.safetensors.index.json").exists()

    @staticmethod
    def _has_orbax(path: Path) -> bool:
        """Return ``True`` when *path* contains an Orbax/OCDBT checkpoint."""
        return (path / "ocdbt.process_0").is_dir() and (path / "manifest.ocdbt").exists()

    @classmethod
    def _has_model_files(cls, path: Path) -> bool:
        """Return ``True`` when *path* contains safetensors or OCDBT model files."""
        return cls._has_safetensors(path) or cls._has_orbax(path)

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
        if cached_path.exists() and cached_path.is_dir() and self._has_model_files(cached_path):
            self._ensure_safetensors(cached_path)
            return cached_path

        if download_if_missing:
            return self.download_sync(model_id)

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

    @staticmethod
    def _is_within_cache_root(path: Path, cache_root: Path) -> bool:
        """Return ``True`` when *path* resolves under *cache_root*."""
        try:
            path.resolve(strict=False).relative_to(cache_root.resolve(strict=False))
        except ValueError:
            return False
        return True

    @staticmethod
    def _cleanup_dir(path: Path) -> None:
        """Remove *path* recursively when it exists."""
        if path.exists():
            shutil.rmtree(path)

    def _get_tokenizer_path(self, clean_id: str) -> str | None:
        """Determine the tokenizer path based on model family."""
        if "gemma3n" in clean_id:
            return "tokenizers/tokenizer_gemma3n.model"
        if "gemma3" in clean_id:
            return "tokenizers/tokenizer_gemma3.model"
        if "gemma2" in clean_id:
            return "tokenizers/tokenizer_gemma2.model"
        return None

    def _make_store(self) -> obs.store.GCSStore:
        return obs.store.GCSStore("gemma-data", config={"skip_signature": "true"})  # type: ignore[arg-type]

    @staticmethod
    def _normalize_list_page(page: object) -> list[object]:
        return list(page) if isinstance(page, list) else [page]

    def _list_remote_files_sync(self, store: obs.store.GCSStore, prefix: str, clean_id: str) -> list[str]:
        paths: list[str] = []
        try:
            for page in obs.list(store, prefix):
                for item in self._normalize_list_page(page):
                    # We know item is a dict-like object returned by obstore
                    path = item["path"]  # type: ignore[index]
                    if not path.endswith("_$folder$"):
                        paths.append(path)
        except Exception as exc:
            msg = f"Failed to list model {clean_id} from GCS: {exc}"
            raise self.GCSDownloadError(msg) from exc
        return paths

    async def _list_remote_files_async(self, store: obs.store.GCSStore, prefix: str, clean_id: str) -> list[str]:
        paths: list[str] = []
        try:
            async for page in store.list_async(prefix):
                for item in self._normalize_list_page(page):
                    path = item["path"]  # type: ignore[index]
                    if not path.endswith("_$folder$"):
                        paths.append(path)
        except Exception as exc:
            msg = f"Failed to list model {clean_id} from GCS: {exc}"
            raise self.GCSDownloadError(msg) from exc
        return paths

    @staticmethod
    def _write_file(destination: Path, data: bytes) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

    def _finalize_download(
        self, clean_id: str, local_dir: Path, staging_dir: Path, *, tokenizer_required: bool
    ) -> Path:
        if tokenizer_required and not (staging_dir / "tokenizer.model").exists():
            msg = f"Download failed for '{clean_id}': integrity error (missing tokenizer.model)"
            raise ValueError(msg)
        if not self._has_model_files(staging_dir):
            msg = f"Download failed for '{clean_id}': integrity error (missing model artifacts)"
            raise ValueError(msg)
        if not self._is_within_cache_root(local_dir, self.cache_path):
            msg = f"Downloader returned invalid cache path for '{clean_id}'"
            raise ValueError(msg)
        if local_dir.exists():
            self._cleanup_dir(local_dir)
        staging_dir.rename(local_dir)
        self._ensure_safetensors(local_dir)
        return local_dir

    def download_sync(self, model_id: str) -> Path:
        """Download a model via the obstore native backend."""
        clean_id = self._clean_model_id(model_id)
        local_dir = self._cache_dir_for_model_id(self.cache_path, model_id)
        prefix = f"checkpoints/{clean_id}/"
        store = self._make_store()
        tokenizer_path = self._get_tokenizer_path(clean_id)

        paths_to_download = self._list_remote_files_sync(store, prefix, clean_id)
        if not paths_to_download:
            msg = f"Model '{clean_id}' was not found in the public gemma-data bucket."
            raise self.ModelNotFoundError(msg)
        if tokenizer_path:
            paths_to_download.append(tokenizer_path)

        try:
            logger.info("Downloading %d files for %s from Google Cloud Storage...", len(paths_to_download), model_id)
            staging_dir = Path(tempfile.mkdtemp(prefix=f".{local_dir.name}.", dir=self.cache_path))
            try:
                for remote_path in paths_to_download:
                    result = obs.get(store, remote_path)
                    data = result.bytes().to_bytes()
                    rel_path = "tokenizer.model" if remote_path == tokenizer_path else remote_path.removeprefix(prefix)
                    self._write_file(staging_dir / rel_path, data)
                return self._finalize_download(
                    clean_id, local_dir, staging_dir, tokenizer_required=tokenizer_path is not None
                )
            except Exception:
                self._cleanup_dir(staging_dir)
                raise
        except Exception:
            if local_dir.exists() and not self._has_model_files(local_dir):
                self._cleanup_dir(local_dir)
            raise

    async def download_async(self, model_id: str) -> Path:
        """Download a model via obstore's native async backend."""
        clean_id = self._clean_model_id(model_id)
        local_dir = self._cache_dir_for_model_id(self.cache_path, model_id)
        prefix = f"checkpoints/{clean_id}/"
        store = self._make_store()
        tokenizer_path = self._get_tokenizer_path(clean_id)

        paths_to_download = await self._list_remote_files_async(store, prefix, clean_id)
        if not paths_to_download:
            msg = f"Model '{clean_id}' was not found in the public gemma-data bucket."
            raise self.ModelNotFoundError(msg)
        if tokenizer_path:
            paths_to_download.append(tokenizer_path)

        try:
            logger.info("Downloading %d files for %s from Google Cloud Storage...", len(paths_to_download), model_id)
            staging_dir = Path(tempfile.mkdtemp(prefix=f".{local_dir.name}.", dir=self.cache_path))
            try:
                for remote_path in paths_to_download:
                    result = await obs.get_async(store, remote_path)
                    data = (await result.bytes_async()).to_bytes()
                    rel_path = "tokenizer.model" if remote_path == tokenizer_path else remote_path.removeprefix(prefix)
                    await asyncio.to_thread(self._write_file, staging_dir / rel_path, data)
                return await asyncio.to_thread(
                    self._finalize_download,
                    clean_id,
                    local_dir,
                    staging_dir,
                    tokenizer_required=tokenizer_path is not None,
                )
            except Exception:
                await asyncio.to_thread(self._cleanup_dir, staging_dir)
                raise
        except Exception:
            if local_dir.exists() and not self._has_model_files(local_dir):
                await asyncio.to_thread(self._cleanup_dir, local_dir)
            raise

    async def resolve_model_async(
        self, model_id: str, *, download_if_missing: bool = False, strict: bool = False, **_kwargs: object
    ) -> Path:
        """Resolve a model ID to a local path without blocking the active asyncio event loop."""
        local_path = Path(model_id)

        def _check_local() -> tuple[bool, bool]:
            return local_path.exists(), local_path.is_dir()

        local_exists, local_is_dir = await asyncio.to_thread(_check_local)

        if local_exists and local_is_dir:
            return local_path

        if local_exists and not local_is_dir:
            msg = f"Model path '{model_id}' exists but is not a directory."
            if strict:
                raise ValueError(msg)
            return local_path

        cached_path = self._cache_dir_for_model_id(self.cache_path, model_id)

        def _check_cached() -> bool:
            return cached_path.exists() and cached_path.is_dir() and self._has_model_files(cached_path)

        cached_valid = await asyncio.to_thread(_check_cached)

        if cached_valid:
            await asyncio.to_thread(self._ensure_safetensors, cached_path)
            return cached_path

        if download_if_missing:
            return await self.download_async(model_id)

        if strict:
            msg = (
                f"Cannot resolve model path '{model_id}'. "
                "Use an existing local directory or a valid Google model id (e.g., gemma-3-1b-it)."
            )
            raise ValueError(msg)
        return Path(model_id)

    @classmethod
    def _ensure_safetensors(cls, path: Path) -> None:
        """Convert an Orbax checkpoint to safetensors if needed."""
        if cls._has_safetensors(path):
            return
        if not cls._has_orbax(path):
            return
        from .convert import convert_orbax_to_safetensors  # noqa: PLC0415

        convert_orbax_to_safetensors(path)
