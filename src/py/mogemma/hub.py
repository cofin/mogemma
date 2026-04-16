"""Model resolution and GCS download helpers."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import obstore as obs
from obstore.store import GCSStore, LocalStore

logger = logging.getLogger(__name__)

_GCS_BUCKET = "gemma-data"
_GCS_TOKENIZER_PATH = "tokenizers/tokenizer_gemma4.model"
_ASYNC_DOWNLOAD_CONCURRENCY = 6

KNOWN_GCS_MODELS: frozenset[str] = frozenset({
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "google/gemma-4-26B-A4B-it",
})
"""Model IDs currently published under ``gs://gemma-data/checkpoints/``.

Hand-maintained. A live probe on 2026-04-16 confirmed these three prefixes
resolve to non-empty listings; the pretrained ``E2B`` / ``E4B`` variants are
not yet published and intentionally excluded.

Public only so the test suite can use it as an allow-list. Not enforced at
runtime — arbitrary ``model_path`` values still flow through the resolver
and surface a clean GCS 404 on a typo instead of a ``KeyError``."""


class HubManager:
    """Manages downloading and caching Gemma 4 models from Google Cloud Storage."""

    def __init__(self, cache_path: str | Path | None = None) -> None:
        """Initialize the HubManager."""
        if cache_path is None:
            configured_cache_path = os.getenv("MOGEMMA_CACHE_DIR")
            if configured_cache_path:
                self.cache_path = Path(configured_cache_path)
            else:
                self.cache_path = Path.home() / ".cache" / "mogemma"
        else:
            self.cache_path = Path(cache_path)

        self.cache_path.mkdir(parents=True, exist_ok=True)

    # ── Store factory ──────────────────────────────────────────────────

    @staticmethod
    def _make_gcs_store() -> GCSStore:
        """Create an obstore ``GCSStore`` for the public ``gemma-data`` bucket.

        Uses ``skip_signature=true`` because the bucket is public and requires
        no authentication.
        """
        # ``config`` is typed as ``GCSConfig | None`` in obstore stubs; passing
        # a plain dict matches the documented runtime API, so we cast for type
        # checkers without importing the private typing alias.
        config: Any = {"skip_signature": "true"}
        return GCSStore(_GCS_BUCKET, config=config)

    # ── Model-id helpers ───────────────────────────────────────────────

    @staticmethod
    def _clean_model_id(model_id: str) -> str:
        """Normalize model id to GCS path form: lowercase, no ``google/``, ``gemma-`` → ``gemma``."""
        clean_id = model_id.removeprefix("google/")
        if clean_id.startswith("gemma-"):
            clean_id = clean_id.replace("gemma-", "gemma", 1)
        return clean_id.lower()

    @staticmethod
    def _cache_dir_for_model_id(cache_root: Path, model_id: str) -> Path:
        return cache_root / model_id.replace("/", "--")

    # ── GCS path helpers ───────────────────────────────────────────────

    @staticmethod
    def _gcs_checkpoint_prefix(clean_id: str) -> str:
        """Return the GCS prefix under which *clean_id*'s checkpoint files live."""
        return f"checkpoints/{clean_id}/"

    @staticmethod
    def _gcs_tokenizer_path() -> str:
        """Return the bucket-relative path to the shared Gemma 4 tokenizer."""
        return _GCS_TOKENIZER_PATH

    # ── Local file helpers ─────────────────────────────────────────────

    @staticmethod
    def _get_store_and_path(path: Path | str) -> tuple[LocalStore, str]:
        p = Path(path).resolve()
        return LocalStore("/"), str(p).lstrip("/")

    @staticmethod
    def _head_exists(store: LocalStore | GCSStore, path: str) -> bool:
        try:
            obs.head(store, path)
        except Exception as exc:  # noqa: BLE001
            logger.debug("obstore head failed for %s", path, exc_info=exc)
            return False
        else:
            return True

    @staticmethod
    def _listing_has_entries(store: LocalStore, path: str) -> bool:
        try:
            return any(True for _ in obs.list(store, path))
        except Exception as exc:  # noqa: BLE001
            logger.debug("obstore list failed for %s", path, exc_info=exc)
            return False

    @staticmethod
    def _has_safetensors(path: Path) -> bool:
        """Return ``True`` when *path* contains ready-to-use safetensors files."""
        store, p = HubManager._get_store_and_path(path)
        return HubManager._head_exists(store, f"{p}/model.safetensors") or HubManager._head_exists(
            store, f"{p}/model.safetensors.index.json"
        )

    @staticmethod
    def _has_orbax(path: Path) -> bool:
        """Return ``True`` when *path* contains an Orbax/OCDBT checkpoint."""
        return (path / "ocdbt.process_0").is_dir() and (path / "manifest.ocdbt").exists()

    _ORBAX_ARTIFACT_NAMES: tuple[str, ...] = (
        "ocdbt.process_0",
        "manifest.ocdbt",
        "_METADATA",
        "_CHECKPOINT_METADATA",
        "descriptor",
        "d",
        "commit_success.txt",
    )

    @classmethod
    def _cleanup_orbax_artifacts(cls, path: Path) -> None:
        """Remove Orbax/OCDBT residue under *path* after successful conversion.

        Preserves ``config.json``, tokenizer files, and any safetensors output.
        """
        for name in cls._ORBAX_ARTIFACT_NAMES:
            target = path / name
            if not target.exists():
                continue
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()

    @classmethod
    def _has_model_files(cls, path: Path) -> bool:
        """Return ``True`` when *path* contains safetensors or Orbax model files."""
        return cls._has_safetensors(path) or cls._has_orbax(path)

    # ── Remote file enumeration ────────────────────────────────────────

    @staticmethod
    def _relative_under(prefix: str, full_path: str) -> str | None:
        """Return *full_path* stripped of *prefix*, or ``None`` if it should be skipped."""
        if not full_path.startswith(prefix):
            return None
        rel = full_path[len(prefix) :]
        if not rel:
            return None
        # obstore/GCS can surface pseudo-directory marker objects like "d_$folder$".
        if "_$folder$" in rel:
            return None
        return rel

    @classmethod
    def _list_remote_files(cls, store: GCSStore, prefix: str) -> list[str]:
        """Enumerate all object paths under *prefix* (returned relative to *prefix*)."""
        results: list[str] = []
        for page in obs.list(store, prefix):
            for obj in page:
                rel = cls._relative_under(prefix, obj["path"])
                if rel is not None:
                    results.append(rel)
        return results

    @classmethod
    async def _list_remote_files_async(cls, store: GCSStore, prefix: str) -> list[str]:
        """Async variant of :meth:`_list_remote_files`.

        ``obs.list`` returns a ``ListStream`` that supports both ``__iter__``
        and ``__aiter__``, so the async variant iterates the same stream with
        ``async for``.
        """
        results: list[str] = []
        async for page in obs.list(store, prefix):
            for obj in page:
                rel = cls._relative_under(prefix, obj["path"])
                if rel is not None:
                    results.append(rel)
        return results

    # ── File I/O helpers ───────────────────────────────────────────────

    @staticmethod
    def _write_file(destination: Path, data: bytes) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

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

    # ── Tokenizer resolution ───────────────────────────────────────────

    def _get_tokenizer_path(self, clean_id: str) -> str | None:
        """Return the local tokenizer filename for *clean_id* (after download)."""
        if "gemma4" in clean_id:
            return "tokenizer.model"
        return None

    def resolve_tokenizer(self, model_dir: Path, *, tokenizer_path: Path | None = None) -> Path:
        """Resolve the tokenizer file with priority: explicit > cached > error.

        Args:
            model_dir: The local directory containing downloaded model files.
            tokenizer_path: Optional explicit path provided by the user.

        Returns:
            Path to the tokenizer file.

        Raises:
            FileNotFoundError: If no tokenizer can be found.
        """
        # 1. Explicit user-provided path (highest priority)
        if tokenizer_path is not None:
            if tokenizer_path.exists():
                return tokenizer_path
            msg = f"Explicit tokenizer path does not exist: {tokenizer_path}"
            raise FileNotFoundError(msg)

        # 2. Tokenizer downloaded alongside the model
        local_tokenizer = model_dir / "tokenizer.model"
        if local_tokenizer.exists():
            return local_tokenizer

        msg = (
            f"No tokenizer found in {model_dir}. "
            "Expected 'tokenizer.model' from the GCS download (gs://gemma-data/tokenizers/tokenizer_gemma4.model)."
        )
        raise FileNotFoundError(msg)

    # ── Config validation ──────────────────────────────────────────────

    @staticmethod
    def validate_config_json(config: dict[str, Any]) -> None:
        """Validate that a config.json contains expected Gemma 4 fields.

        Raises:
            ValueError: If required fields are missing or model_type is not Gemma 4.
        """
        model_type = config.get("model_type")
        if model_type is None:
            msg = "config.json missing required field 'model_type'"
            raise ValueError(msg)

        if not model_type.startswith("gemma4"):
            msg = f"Expected a Gemma 4 model (model_type starting with 'gemma4'), got '{model_type}'"
            raise ValueError(msg)

        if "num_hidden_layers" not in config:
            msg = "config.json missing required field 'num_hidden_layers'"
            raise ValueError(msg)

    # ── Error types ────────────────────────────────────────────────────

    class GCSDownloadError(ConnectionError):
        """Raised when a download from the ``gemma-data`` GCS bucket fails."""

    class ModelNotFoundError(FileNotFoundError):
        """Raised when a model checkpoint is not found in the ``gemma-data`` GCS bucket."""

    # ── Resolve ────────────────────────────────────────────────────────

    def resolve_model(
        self, model_id: str, *, download_if_missing: bool = False, strict: bool = False, **_kwargs: object
    ) -> Path:
        """Resolve a model ID to a local path."""
        local_path = Path(model_id)
        store, p = self._get_store_and_path(local_path)
        is_dir = self._listing_has_entries(store, p)

        if is_dir:
            return local_path

        exists = self._head_exists(store, p)

        if exists:
            if local_path.is_file() and local_path.suffix == ".safetensors":
                return local_path
            msg = f"Model path '{model_id}' exists but is not a directory."
            if strict:
                raise ValueError(msg)
            return local_path

        cached_path = self._cache_dir_for_model_id(self.cache_path, model_id)
        if self._has_model_files(cached_path):
            return cached_path

        if download_if_missing:
            return self.download_sync(model_id)

        if strict:
            msg = (
                f"Cannot resolve model path '{model_id}'. "
                "Use an existing local directory or a Gemma 4 model id that maps to the "
                f"public gs://{_GCS_BUCKET} bucket (e.g., google/gemma-4-26B-A4B-it)."
            )
            raise ValueError(msg)

        return Path(model_id)

    # ── Download helpers ───────────────────────────────────────────────

    def _download_file(self, store: GCSStore, remote_path: str, dest: Path) -> None:
        """Download a single object from GCS to *dest*."""
        result = obs.get(store, remote_path)
        data = bytes(result.bytes())
        self._write_file(dest, data)

    async def _download_file_async(self, store: GCSStore, remote_path: str, dest: Path) -> None:
        """Async variant of :meth:`_download_file`."""
        result = await obs.get_async(store, remote_path)
        data = bytes(await result.bytes_async())
        await asyncio.to_thread(self._write_file, dest, data)

    def _finalize_download(
        self, clean_id: str, local_dir: Path, staging_dir: Path, *, tokenizer_required: bool
    ) -> Path:
        if tokenizer_required and not (staging_dir / "tokenizer.model").exists():
            msg = f"Download failed for '{clean_id}': integrity error (missing tokenizer.model)"
            raise ValueError(msg)
        if not self._has_model_files(staging_dir):
            msg = (
                f"Download failed for '{clean_id}': integrity error "
                "(no safetensors or Orbax artifacts in staging directory)"
            )
            raise ValueError(msg)
        if not self._is_within_cache_root(local_dir, self.cache_path):
            msg = f"Downloader returned invalid cache path for '{clean_id}'"
            raise ValueError(msg)
        if local_dir.exists():
            self._cleanup_dir(local_dir)
        staging_dir.rename(local_dir)

        # If the downloaded checkpoint is Orbax-only, convert it to safetensors
        # and drop the Orbax artifacts. On conversion failure the Orbax layout
        # is preserved so the user can retry without re-downloading.
        if self._has_orbax(local_dir) and not self._has_safetensors(local_dir):
            from mogemma.convert import convert_orbax_to_safetensors  # noqa: PLC0415

            convert_orbax_to_safetensors(local_dir)
            if self._has_safetensors(local_dir):
                self._cleanup_orbax_artifacts(local_dir)

        return local_dir

    # ── Download (sync) ────────────────────────────────────────────────

    def download_sync(self, model_id: str) -> Path:
        """Download a Gemma 4 model from the public ``gemma-data`` GCS bucket."""
        clean_id = self._clean_model_id(model_id)
        local_dir = self._cache_dir_for_model_id(self.cache_path, model_id)
        store = self._make_gcs_store()
        tokenizer_local_name = self._get_tokenizer_path(clean_id)
        checkpoint_prefix = self._gcs_checkpoint_prefix(clean_id)

        try:
            remote_files = self._list_remote_files(store, checkpoint_prefix)
        except Exception as exc:
            msg = f"Failed to list GCS checkpoint '{checkpoint_prefix}' for '{model_id}': {exc}"
            raise self.ModelNotFoundError(msg) from exc

        if not remote_files:
            msg = f"No objects found under gs://{_GCS_BUCKET}/{checkpoint_prefix} for '{model_id}'"
            raise self.ModelNotFoundError(msg)

        staging_dir = Path(tempfile.mkdtemp(prefix=f".{local_dir.name}.", dir=self.cache_path))
        try:
            logger.info(
                "Downloading %d checkpoint files for %s from gs://%s/%s ...",
                len(remote_files),
                model_id,
                _GCS_BUCKET,
                checkpoint_prefix,
            )
            for rel_path in remote_files:
                self._download_file(store, checkpoint_prefix + rel_path, staging_dir / rel_path)

            if tokenizer_local_name:
                self._download_file(store, self._gcs_tokenizer_path(), staging_dir / tokenizer_local_name)

            return self._finalize_download(
                clean_id, local_dir, staging_dir, tokenizer_required=tokenizer_local_name is not None
            )
        except Exception as exc:
            self._cleanup_dir(staging_dir)
            if local_dir.exists() and not self._has_model_files(local_dir):
                self._cleanup_dir(local_dir)
            if isinstance(exc, self.ModelNotFoundError | ValueError):
                raise
            msg = f"GCS download failed for '{model_id}': {exc}"
            raise self.GCSDownloadError(msg) from exc

    # ── Download (async) ───────────────────────────────────────────────

    async def download_async(self, model_id: str) -> Path:
        """Download a Gemma 4 model from the public ``gemma-data`` GCS bucket (async)."""
        clean_id = self._clean_model_id(model_id)
        local_dir = self._cache_dir_for_model_id(self.cache_path, model_id)
        store = self._make_gcs_store()
        tokenizer_local_name = self._get_tokenizer_path(clean_id)
        checkpoint_prefix = self._gcs_checkpoint_prefix(clean_id)

        try:
            remote_files = await self._list_remote_files_async(store, checkpoint_prefix)
        except Exception as exc:
            msg = f"Failed to list GCS checkpoint '{checkpoint_prefix}' for '{model_id}': {exc}"
            raise self.ModelNotFoundError(msg) from exc

        if not remote_files:
            msg = f"No objects found under gs://{_GCS_BUCKET}/{checkpoint_prefix} for '{model_id}'"
            raise self.ModelNotFoundError(msg)

        staging_dir = await asyncio.to_thread(
            lambda: Path(tempfile.mkdtemp(prefix=f".{local_dir.name}.", dir=self.cache_path))
        )
        semaphore = asyncio.Semaphore(_ASYNC_DOWNLOAD_CONCURRENCY)

        async def _bounded_download(remote: str, dest: Path) -> None:
            async with semaphore:
                await self._download_file_async(store, remote, dest)

        try:
            logger.info(
                "Downloading %d checkpoint files for %s from gs://%s/%s (concurrency=%d) ...",
                len(remote_files),
                model_id,
                _GCS_BUCKET,
                checkpoint_prefix,
                _ASYNC_DOWNLOAD_CONCURRENCY,
            )
            tasks = [_bounded_download(checkpoint_prefix + rel, staging_dir / rel) for rel in remote_files]
            if tokenizer_local_name:
                tasks.append(_bounded_download(self._gcs_tokenizer_path(), staging_dir / tokenizer_local_name))
            await asyncio.gather(*tasks)

            return await asyncio.to_thread(
                self._finalize_download,
                clean_id,
                local_dir,
                staging_dir,
                tokenizer_required=tokenizer_local_name is not None,
            )
        except Exception as exc:
            await asyncio.to_thread(self._cleanup_dir, staging_dir)
            if local_dir.exists() and not self._has_model_files(local_dir):
                await asyncio.to_thread(self._cleanup_dir, local_dir)
            if isinstance(exc, self.ModelNotFoundError | ValueError):
                raise
            msg = f"GCS download failed for '{model_id}': {exc}"
            raise self.GCSDownloadError(msg) from exc

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
            return cached_path

        if download_if_missing:
            return await self.download_async(model_id)

        if strict:
            msg = (
                f"Cannot resolve model path '{model_id}'. "
                "Use an existing local directory or a Gemma 4 model id that maps to the "
                f"public gs://{_GCS_BUCKET} bucket (e.g., google/gemma-4-26B-A4B-it)."
            )
            raise ValueError(msg)
        return Path(model_id)
