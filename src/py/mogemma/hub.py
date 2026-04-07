"""Model resolution and HuggingFace download helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import obstore as obs
from obstore.store import HTTPStore, LocalStore

logger = logging.getLogger(__name__)

_HF_BASE = "https://huggingface.co"


class HubManager:
    """Manages downloading and caching Gemma 4 models from HuggingFace."""

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

    # ── URL helpers ────────────────────────────────────────────────────

    @staticmethod
    def _hf_resolve_url(repo_id: str, filename: str) -> str:
        """Build a HuggingFace ``/resolve/main/`` URL for *filename*."""
        return f"{_HF_BASE}/{repo_id}/resolve/main/{filename}"

    # ── Store factory ──────────────────────────────────────────────────

    @staticmethod
    def _make_hf_store(repo_id: str, token: str | None = None) -> HTTPStore:
        """Create an obstore HTTPStore pointed at a HuggingFace repo."""
        base_url = f"{_HF_BASE}/{repo_id}/resolve/main/"
        client_options: dict[str, Any] = {}
        if token:
            client_options["default_headers"] = {"Authorization": f"Bearer {token}"}
        return HTTPStore.from_url(base_url, client_options=client_options)  # type: ignore[arg-type]

    @staticmethod
    def _get_hf_token() -> str | None:
        """Read the ``HF_TOKEN`` environment variable."""
        return os.environ.get("HF_TOKEN")

    # ── Model-id helpers ───────────────────────────────────────────────

    @staticmethod
    def _clean_model_id(model_id: str) -> str:
        """Normalize model id (strip ``google/`` prefix, collapse ``gemma-`` → ``gemma``)."""
        clean_id = model_id.removeprefix("google/")
        return clean_id.replace("gemma-", "gemma") if clean_id.startswith("gemma-") else clean_id

    @staticmethod
    def _cache_dir_for_model_id(cache_root: Path, model_id: str) -> Path:
        return cache_root / model_id.replace("/", "--")

    # ── Local file helpers ─────────────────────────────────────────────

    @staticmethod
    def _get_store_and_path(path: Path | str) -> tuple[LocalStore, str]:
        p = Path(path).resolve()
        return LocalStore("/"), str(p).lstrip("/")

    @staticmethod
    def _head_exists(store: LocalStore | HTTPStore, path: str) -> bool:
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

    @classmethod
    def _has_model_files(cls, path: Path) -> bool:
        """Return ``True`` when *path* contains safetensors model files."""
        return cls._has_safetensors(path)

    # ── Shard discovery ────────────────────────────────────────────────

    @staticmethod
    def _parse_shard_filenames(index: dict[str, Any]) -> list[str]:
        """Extract unique shard filenames from a ``model.safetensors.index.json`` weight_map."""
        weight_map = index.get("weight_map")
        if not weight_map:
            msg = "Index JSON missing or empty 'weight_map'"
            raise ValueError(msg)
        return sorted(set(weight_map.values()))

    # ── Download skip logic ────────────────────────────────────────────

    @staticmethod
    def _should_skip_download(dest: Path, expected_size: int | None) -> bool:
        """Return ``True`` when *dest* exists with the expected byte size."""
        if expected_size is None:
            return False
        if not dest.exists():
            return False
        return dest.stat().st_size == expected_size

    # ── File I/O helpers ───────────────────────────────────────────────

    @staticmethod
    def _write_file(destination: Path, data: bytes) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

    @staticmethod
    def _normalize_list_page(page: object) -> list[object]:
        return list(page) if isinstance(page, list) else [page]

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
        """Determine the tokenizer filename for download based on model family."""
        if "gemma4" in clean_id:
            return "tokenizer.model"
        return None

    def resolve_tokenizer(self, model_dir: Path, *, tokenizer_path: Path | None = None) -> Path:
        """Resolve the tokenizer file with priority: explicit > HF local > error.

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

        # 2. HF-downloaded tokenizer.model in model directory
        hf_tokenizer = model_dir / "tokenizer.model"
        if hf_tokenizer.exists():
            return hf_tokenizer

        msg = f"No tokenizer found in {model_dir}. Expected 'tokenizer.model' from HuggingFace download."
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

    class HFDownloadError(ConnectionError):
        """Raised when a HuggingFace download fails."""

    class ModelNotFoundError(FileNotFoundError):
        """Raised when a model is not found on HuggingFace."""

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
                "Use an existing local directory or a valid HuggingFace model id (e.g., google/gemma-4-31B-it)."
            )
            raise ValueError(msg)

        return Path(model_id)

    # ── Download (sync) ────────────────────────────────────────────────

    def _fetch_index_json(self, store: HTTPStore, repo_id: str) -> dict[str, Any]:
        """Fetch and parse ``model.safetensors.index.json`` from HuggingFace."""
        try:
            result = obs.get(store, "model.safetensors.index.json")
            return dict(json.loads(bytes(result.bytes())))
        except Exception as exc:
            msg = f"Failed to fetch model index for '{repo_id}' from HuggingFace: {exc}"
            raise self.ModelNotFoundError(msg) from exc

    async def _fetch_index_json_async(self, store: HTTPStore, repo_id: str) -> dict[str, Any]:
        """Fetch and parse ``model.safetensors.index.json`` from HuggingFace (async)."""
        try:
            result = await obs.get_async(store, "model.safetensors.index.json")
            return dict(json.loads(bytes(await result.bytes_async())))
        except Exception as exc:
            msg = f"Failed to fetch model index for '{repo_id}' from HuggingFace: {exc}"
            raise self.ModelNotFoundError(msg) from exc

    def _download_hf_file(
        self, store: HTTPStore, filename: str, dest_dir: Path, expected_size: int | None = None
    ) -> None:
        """Download a single file from HuggingFace to *dest_dir*."""
        dest = dest_dir / filename
        if self._should_skip_download(dest, expected_size):
            logger.debug("Skipping %s (already exists with correct size)", filename)
            return
        result = obs.get(store, filename)
        data = bytes(result.bytes())
        self._write_file(dest, data)

    async def _download_hf_file_async(
        self, store: HTTPStore, filename: str, dest_dir: Path, expected_size: int | None = None
    ) -> None:
        """Download a single file from HuggingFace (async)."""
        dest = dest_dir / filename
        if self._should_skip_download(dest, expected_size):
            logger.debug("Skipping %s (already exists with correct size)", filename)
            return
        result = await obs.get_async(store, filename)
        data = bytes(await result.bytes_async())
        await asyncio.to_thread(self._write_file, dest, data)

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
        return local_dir

    def download_sync(self, model_id: str) -> Path:
        """Download a Gemma 4 model from HuggingFace."""
        clean_id = self._clean_model_id(model_id)
        local_dir = self._cache_dir_for_model_id(self.cache_path, model_id)
        token = self._get_hf_token()
        store = self._make_hf_store(model_id, token=token)
        tokenizer_path = self._get_tokenizer_path(clean_id)

        # 1. Fetch index.json to discover shard files
        index = self._fetch_index_json(store, model_id)
        shard_files = self._parse_shard_filenames(index)

        # 2. Build download list: index.json + shards + config.json + tokenizer
        files_to_download = ["model.safetensors.index.json", "config.json", *shard_files]
        if tokenizer_path:
            files_to_download.append(tokenizer_path)

        try:
            logger.info("Downloading %d files for %s from HuggingFace...", len(files_to_download), model_id)
            staging_dir = Path(tempfile.mkdtemp(prefix=f".{local_dir.name}.", dir=self.cache_path))
            try:
                # Write the index.json we already fetched
                self._write_file(staging_dir / "model.safetensors.index.json", json.dumps(index).encode())

                # Download remaining files
                for filename in files_to_download:
                    if filename == "model.safetensors.index.json":
                        continue  # Already written above
                    self._download_hf_file(store, filename, staging_dir)

                return self._finalize_download(
                    clean_id, local_dir, staging_dir, tokenizer_required=tokenizer_path is not None
                )
            except Exception:
                self._cleanup_dir(staging_dir)
                raise
        except self.ModelNotFoundError:
            raise
        except Exception:
            if local_dir.exists() and not self._has_model_files(local_dir):
                self._cleanup_dir(local_dir)
            raise

    async def download_async(self, model_id: str) -> Path:
        """Download a Gemma 4 model from HuggingFace (async)."""
        clean_id = self._clean_model_id(model_id)
        local_dir = self._cache_dir_for_model_id(self.cache_path, model_id)
        token = self._get_hf_token()
        store = self._make_hf_store(model_id, token=token)
        tokenizer_path = self._get_tokenizer_path(clean_id)

        # 1. Fetch index.json
        index = await self._fetch_index_json_async(store, model_id)
        shard_files = self._parse_shard_filenames(index)

        # 2. Build download list
        files_to_download = ["model.safetensors.index.json", "config.json", *shard_files]
        if tokenizer_path:
            files_to_download.append(tokenizer_path)

        try:
            logger.info("Downloading %d files for %s from HuggingFace...", len(files_to_download), model_id)
            staging_dir = Path(tempfile.mkdtemp(prefix=f".{local_dir.name}.", dir=self.cache_path))
            try:
                # Write the index.json we already fetched
                await asyncio.to_thread(
                    self._write_file, staging_dir / "model.safetensors.index.json", json.dumps(index).encode()
                )

                # Download remaining files
                for filename in files_to_download:
                    if filename == "model.safetensors.index.json":
                        continue
                    await self._download_hf_file_async(store, filename, staging_dir)

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
        except self.ModelNotFoundError:
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
            return cached_path

        if download_if_missing:
            return await self.download_async(model_id)

        if strict:
            msg = (
                f"Cannot resolve model path '{model_id}'. "
                "Use an existing local directory or a valid HuggingFace model id (e.g., google/gemma-4-31B-it)."
            )
            raise ValueError(msg)
        return Path(model_id)
