# Learnings

## [2026-02-25] - pypi-cibuildwheel_20260225: Linux Mojo Compiler requirements
- **Files changed:** `pyproject.toml`, `.github/workflows/publish.yml`, `.github/workflows/test-build.yml`
- **Learning:** The `mojo` pip package (`mojo-0.26.1.0-py3-none-manylinux_2_34_x86_64.whl`) requires `GLIBCXX_3.4.30` (from GCC 12/libstdc++). However, `cibuildwheel`'s official `manylinux_2_34_x86_64` image is based on AlmaLinux 9, which only provides `GLIBCXX_3.4.29` (GCC 11). As a result, invoking the `mojo` compiler inside the `manylinux_2_34` container fails with `/lib64/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`.
- **Gotcha:** This prevents successful building of `manylinux` wheels for Mojo extensions using standard `cibuildwheel` manylinux images until a `manylinux` image with GCC 12+ is used or Modular fixes the wheel to depend on older `libstdc++`.
