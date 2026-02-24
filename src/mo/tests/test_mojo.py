import subprocess
import pytest
from pathlib import Path

MO_TESTS_DIR = Path(__file__).parent

@pytest.mark.parametrize("test_file", [
    "test_layers.mojo",
    "test_model.mojo",
    "test_ops.mojo",
    "test_nano_layers.mojo",
    "test_altup_contract.mojo",
])
def test_mojo_unit_tests(test_file):
    test_path = MO_TESTS_DIR / test_file
    # Use -I src/mo to include the mogemma module
    result = subprocess.run(
        ["mojo", "-I", str(MO_TESTS_DIR.parent), str(test_path)],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Mojo test {test_file} failed:\n{result.stdout}\n{result.stderr}"
