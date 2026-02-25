from pathlib import Path


def test_mojo_sources_live_in_package_namespace() -> None:
    root = Path(__file__).resolve().parents[3]

    for name in ("core.mojo", "layers.mojo", "model.mojo", "ops.mojo"):
        assert (root / "src" / "mo" / "mogemma" / name).is_file()
        assert not (root / "src" / "mo" / name).exists()

    assert (root / "src" / "mo" / "mogemma" / "__init__.mojo").is_file()


def test_build_targets_point_to_namespaced_core() -> None:
    root = Path(__file__).resolve().parents[3]
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    hatch_build = (root / "tools" / "hatch_build.py").read_text(encoding="utf-8")

    assert "src/mo/mogemma/core.mojo" in makefile
    assert 'root / "src" / "mo" / "mogemma" / "core.mojo"' in hatch_build
