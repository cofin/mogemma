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
        # Create a valid minimal metadata dict to avoid Mojo init aborting
        t1_meta = metadata["my_tensor"]
        valid_metadata = {
            "model.embed_tokens.weight": t1_meta,
            "model.norm.weight": t1_meta,
            "lm_head.weight": t1_meta,
            "model.layers.0.input_layernorm.weight": t1_meta,
            "model.layers.0.post_attention_layernorm.weight": t1_meta,
            "model.layers.0.self_attn.q_proj.weight": t1_meta,
            "model.layers.0.self_attn.k_proj.weight": (t1_meta[0], (512, 1024), t1_meta[2]),
            "model.layers.0.self_attn.v_proj.weight": t1_meta,
            "model.layers.0.self_attn.o_proj.weight": t1_meta,
            "model.layers.0.mlp.gate_proj.weight": t1_meta,
            "model.layers.0.mlp.up_proj.weight": t1_meta,
            "model.layers.0.mlp.down_proj.weight": t1_meta,
        }

        result = _core.init_model(valid_metadata)

        assert "engine" in result
        assert "runtime" in result
        assert result["engine"] == "Mojo Pure Inference Engine"
        assert result["arch"] == "standard"


def test_loader_rejects_missing_path() -> None:
    missing_path = Path("/does/not/exist")
    with pytest.raises(Exception, match=r"No model.safetensors"):
        # Accessing .stat() or .is_file() on a nonexistent path might not raise,
        # but our loader or Path handles it. Wait, SafetensorsLoader checks .exists()
        SafetensorsLoader(missing_path)


def test_loader_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(Exception, match=r"No model.safetensors or index found"):
        SafetensorsLoader(tmp_path)
