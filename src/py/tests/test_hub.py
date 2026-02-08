import pytest
from pathlib import Path
from mogemma.hub import HubManager

def test_hub_manager_default_path():
    """Verify default cache path is in user home."""
    hub = HubManager()
    assert ".cache/mogemma" in str(hub.cache_path)

def test_hub_manager_custom_path(tmp_path):
    """Verify custom cache path is respected."""
    hub = HubManager(cache_path=tmp_path)
    assert hub.cache_path == tmp_path

def test_resolve_model_path_local(tmp_path):
    """Verify it returns local path if it exists."""
    model_dir = tmp_path / "gemma-3-4b"
    model_dir.mkdir()
    hub = HubManager(cache_path=tmp_path)
    
    resolved = hub.resolve_model("gemma-3-4b")
    assert resolved == model_dir

def test_resolve_model_hf_id(tmp_path):
    """Verify it returns cached path for HF ID."""
    hub = HubManager(cache_path=tmp_path)
    model_id = "google/gemma-3-4b-it"
    resolved = hub.resolve_model(model_id)
    expected = tmp_path / "google--gemma-3-4b-it"
    assert resolved == expected

@pytest.mark.skip(reason="Requires network/HF token")
def test_download_model_from_hub():
    """Placeholder for hub download test."""
    pass
