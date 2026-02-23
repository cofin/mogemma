from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from mogemma import _core
from mogemma.loader import SafetensorsLoader

EXPECTED_BYTE_VALUE = 42


@pytest.fixture
def sample_safetensors(tmp_path: Path) -> Path:
    """Create a sample safetensors file for testing."""
    file_path = tmp_path / "test_model.safetensors"
    # Create some dummy weights
    t1 = np.array([EXPECTED_BYTE_VALUE, 1, 2, 3], dtype=np.uint8)
    t2 = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    save_file({"my_tensor": t1, "other_tensor": t2}, str(file_path))
    return file_path


def test_loader_zero_copy_bridge(sample_safetensors: Path) -> None:
    """Test that we can extract memory pointers and pass them to Mojo."""
    with SafetensorsLoader(sample_safetensors) as loader:
        metadata = loader.get_tensor_metadata()

        assert "my_tensor" in metadata
        assert "other_tensor" in metadata

        ptr, shape, dtype = metadata["my_tensor"]
        assert isinstance(ptr, int)
        assert shape == (4,)
        assert dtype == "U8"

        # Test the Mojo bridge
        # Our current init_model_mojo grabs the first key, but dictionary order
        # in Mojo might differ, so we'll pass a dict with just one key
        single_metadata = {"my_tensor": metadata["my_tensor"]}
        result = _core.init_model(single_metadata)

        assert "first_byte" in result
        assert result["first_byte"] == EXPECTED_BYTE_VALUE, "Memory mismatch in Mojo zero-copy bridge!"


def test_loader_rejects_missing_path() -> None:
    missing_path = Path("/does/not/exist")
    with pytest.raises(Exception, match=r"No model.safetensors"):
        # Accessing .stat() or .is_file() on a nonexistent path might not raise,
        # but our loader or Path handles it. Wait, SafetensorsLoader checks .exists()
        SafetensorsLoader(missing_path)


def test_loader_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(Exception, match=r"No model.safetensors or index found"):
        SafetensorsLoader(tmp_path)
