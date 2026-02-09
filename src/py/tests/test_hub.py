from pathlib import Path

import pytest

from mogemma.hub import HubManager


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
    model_dir.mkdir()
    hub = HubManager(cache_path=tmp_path)

    resolved = hub.resolve_model(str(model_dir))
    assert resolved == model_dir


def test_resolve_model_cached_path(tmp_path: Path) -> None:
    """Verify cached model directories are returned as filesystem paths."""
    model_id = "google/gemma-3-4b-it"
    cached_dir = tmp_path / "google--gemma-3-4b-it"
    cached_dir.mkdir()
    hub = HubManager(cache_path=tmp_path)

    resolved = hub.resolve_model(model_id)
    assert resolved == cached_dir


def test_resolve_model_hf_id_cache_miss_returns_model_id(tmp_path: Path) -> None:
    """Verify cache misses preserve the hub ID for direct hub resolution."""
    hub = HubManager(cache_path=tmp_path)
    model_id = "google/gemma-3-4b-it"

    resolved = hub.resolve_model(model_id)
    assert resolved == Path(model_id)


@pytest.mark.skip(reason="Requires network/HF token")
def test_download_model_from_hub() -> None:
    """Placeholder for hub download test."""
