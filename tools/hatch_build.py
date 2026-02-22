import os
import subprocess
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Custom build hook to compile Mojo source into a shared library."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Called before the build starts."""
        if self.target_name != "wheel":
            return

        if os.environ.get("MOGEMMA_SKIP_MOJO") == "1":
            print("Skipping Mojo compilation (MOGEMMA_SKIP_MOJO=1)")
            return

        root = Path(self.root)
        mojo_src = root / "src" / "mo" / "core.mojo"
        so_dest = root / "src" / "py" / "mogemma" / "_core.so"

        mojo_bin = self._find_mojo(root)
        print(f"Building Mojo core from {mojo_src} to {so_dest}...")
        so_dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            [str(mojo_bin), "build", "--emit", "shared-lib", str(mojo_src), "-o", str(so_dest)],
            cwd=str(root),
        )
        print(f"Successfully built {so_dest.name}")

    @staticmethod
    def _find_mojo(root: Path) -> str:
        """Locate the mojo compiler binary."""
        import shutil

        path = shutil.which("mojo")
        if path:
            return path
        venv_mojo = root / ".venv" / "bin" / "mojo"
        if venv_mojo.exists():
            return str(venv_mojo)
        msg = "mojo executable not found in PATH or .venv/bin/"
        raise FileNotFoundError(msg)
