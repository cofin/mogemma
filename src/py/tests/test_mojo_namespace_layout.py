from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def test_mojo_sources_live_in_package_namespace() -> None:
    root = Path(__file__).resolve().parents[3]

    for name in ("core.mojo", "layers.mojo", "model.mojo", "ops.mojo"):
        assert (root / "src" / "mo" / "mogemma" / name).is_file()
        assert not (root / "src" / "mo" / name).exists()

    assert (root / "src" / "mo" / "mogemma" / "__init__.mojo").is_file()


def test_build_targets_point_to_namespaced_core() -> None:
    root = Path(__file__).resolve().parents[3]

    with (root / "pyproject.toml").open("rb") as f:
        config = tomllib.load(f)

    jobs = config["tool"]["hatch"]["build"]["targets"]["wheel"]["hooks"]["mojo"]["jobs"]
    core_job = next(j for j in jobs if j["name"] == "core")

    assert core_job["input"] == "src/mo/mogemma/core.mojo"
    assert core_job["module"] == "mogemma._core"
    assert "src/mo" in core_job["include-dirs"]


def test_linux_wheel_repair_retags_as_manylinux() -> None:
    root = Path(__file__).resolve().parents[3]

    with (root / "pyproject.toml").open("rb") as f:
        config = tomllib.load(f)

    linux_config = config["tool"]["cibuildwheel"]["linux"]
    cmd = linux_config["repair-wheel-command"]
    assert "manylinux_2_34" in cmd
    assert "wheel tags" in cmd
