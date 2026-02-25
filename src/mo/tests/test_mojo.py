import os
import subprocess
from pathlib import Path

import pytest

MO_TESTS_DIR = Path(__file__).parent
MOJO_TEST_TIMEOUT_SECONDS = int(os.getenv("MOGEMMA_MOJO_TEST_TIMEOUT_SECONDS", "90"))
RUN_UNSTABLE_MOJO_TESTS = os.getenv("MOGEMMA_RUN_UNSTABLE_MOJO_TESTS", "0") == "1"
UNSTABLE_MOJO_TESTS = {
    "test_layers.mojo",
    "test_nano_layers.mojo",
}

@pytest.mark.parametrize("test_file", [
    "test_layers.mojo",
    "test_model.mojo",
    "test_ops.mojo",
    "test_nano_layers.mojo",
    "test_altup_contract.mojo",
])
def test_mojo_unit_tests(test_file):
    if test_file in UNSTABLE_MOJO_TESTS and not RUN_UNSTABLE_MOJO_TESTS:
        pytest.skip(
            "Known unstable on Mojo 0.26.1 in CI/local (hang/crash). "
            "Set MOGEMMA_RUN_UNSTABLE_MOJO_TESTS=1 to run explicitly."
        )

    test_path = MO_TESTS_DIR / test_file
    cmd = ["mojo", "-I", str(MO_TESTS_DIR.parent), str(test_path)]
    try:
        # Use -I src/mo to include the mogemma module.
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=MOJO_TEST_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        pytest.fail(
            f"Mojo test {test_file} timed out after {MOJO_TEST_TIMEOUT_SECONDS}s.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )

    assert result.returncode == 0, f"Mojo test {test_file} failed:\n{result.stdout}\n{result.stderr}"
