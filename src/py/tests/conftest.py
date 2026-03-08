"""Pytest configuration and global fixtures for mogemma test suite."""

from __future__ import annotations

import ctypes
import importlib.util
from pathlib import Path


def _preload_modular_runtime() -> None:
    spec = importlib.util.find_spec("modular")
    if spec is None or spec.submodule_search_locations is None:
        return

    for location in spec.submodule_search_locations:
        lib_path = Path(location) / "lib" / "libKGENCompilerRTShared.so"
        if not lib_path.exists():
            continue
        try:
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        return


_preload_modular_runtime()
