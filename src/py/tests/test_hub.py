import json
import struct
from pathlib import Path

import pytest

from mogemma.hub import HubManager


def _create_dummy_safetensors(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "model.safetensors").open("wb") as f:
        h = json.dumps({}).encode("utf-8")
        f.write(struct.pack("<Q", len(h)) + h)


def test_hub_manager_default_path() -> None:
    """Verify default cache path is in user home."""
    hub = HubManager()
    assert ".cache/mogemma" in str(hub.cache_path)


def test_hub_manager_custom_path(tmp_path: Path) -> None:
    """Verify custom cache path is respected."""
    hub = HubManager(cache_path=tmp_path)
    assert hub.cache_path == tmp_path


def test_resolve_model_path_local(tmp_path: Path) -> None:
    """Verify local model directories are returned directly."""
    model_dir = tmp_path / "gemma-3-4b"
    _create_dummy_safetensors(model_dir)
    hub = HubManager(cache_path=tmp_path)

    resolved = hub.resolve_model(str(model_dir))
    assert resolved == model_dir


def test_resolve_model_cached_path(tmp_path: Path) -> None:
    """Verify cached model directories are returned as filesystem paths."""
    model_id = "gemma-3-4b-it"
    cached_dir = tmp_path / "gemma-3-4b-it"
    _create_dummy_safetensors(cached_dir)

    hub = HubManager(cache_path=tmp_path)

    resolved = hub.resolve_model(model_id)
    assert resolved == cached_dir


def test_resolve_model_cached_path_ocdbt(tmp_path: Path) -> None:
    """Verify cached OCDBT checkpoint directories are recognized."""
    model_id = "gemma3n-e2b-it"
    cached_dir = tmp_path / "gemma3n-e2b-it"
    cached_dir.mkdir()
    (cached_dir / "manifest.ocdbt").touch()
    (cached_dir / "ocdbt.process_0").mkdir()

    hub = HubManager(cache_path=tmp_path)

    resolved = hub.resolve_model(model_id)
    assert resolved == cached_dir


def test_resolve_model_ignores_stale_cache(tmp_path: Path) -> None:
    """Cache dir with no recognized model files should not be treated as valid."""
    model_id = "gemma-3-4b-it"
    cached_dir = tmp_path / "gemma-3-4b-it"
    cached_dir.mkdir()
    (cached_dir / "tokenizer.model").touch()  # no safetensors or OCDBT

    hub = HubManager(cache_path=tmp_path)

    resolved = hub.resolve_model(model_id, strict=False)
    # Should fall through (not return cached_dir)
    assert resolved != cached_dir


def test_resolve_model_strict_rejects_missing_local_path(tmp_path: Path) -> None:
    """Strict mode with download_if_missing=True will try to download and fail if missing in GCS."""
    hub = HubManager(cache_path=tmp_path)

    with pytest.raises(FileNotFoundError, match="not found in the public gemma-data bucket"):
        hub.resolve_model("bert-base-uncased-missing", download_if_missing=True, strict=True)


def test_clean_model_id() -> None:
    """Verify Google Cloud model id formatting rules."""
    hub = HubManager()
    assert hub._clean_model_id("gemma-3-1b-it") == "gemma3-1b-it"
    assert hub._clean_model_id("google/gemma-3-1b-it") == "gemma3-1b-it"
    assert hub._clean_model_id("gemma3-1b-it") == "gemma3-1b-it"
