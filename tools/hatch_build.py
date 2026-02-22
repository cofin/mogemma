import os
import subprocess
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Custom build hook to compile Mojo source into a shared library."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Called before the build starts."""
        root = Path(self.root)
        mojo_src = root / "src" / "mo" / "core.mojo"
        so_dest = root / "src" / "py" / "mogemma" / "_core.so"

        # Only build for wheels or if specifically requested, but safe to just build it always during initialize
        if self.target_name != "wheel":
            return

        print(f"Building Mojo core from {mojo_src} to {so_dest}...")

        # Ensure the destination directory exists
        so_dest.parent.mkdir(parents=True, exist_ok=True)

        # Run the mojo build command
        try:
            # First try using the `mojo` command directly if available
            subprocess.check_call(
                ["mojo", "build", "--emit", "shared-lib", str(mojo_src), "-o", str(so_dest)],
                cwd=str(root)
            )
        except FileNotFoundError:
            # Fallback to `uv run mojo` if in a uv environment and mojo isn't in PATH
            print("mojo command not found in PATH, trying with `uv run mojo`")
            subprocess.check_call(
                ["uv", "run", "mojo", "build", "--emit", "shared-lib", str(mojo_src), "-o", str(so_dest)],
                cwd=str(root)
            )

        print(f"Successfully built {so_dest.name}")
