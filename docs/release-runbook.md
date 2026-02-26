# Release runbook (v0.1.0)

This runbook defines the concrete evidence and execution flow for the first production release.

## Scope and supported build matrix

- Linux and macOS wheels are built by `.github/workflows/publish.yml`.
- Source distribution is built and published with the same workflow.
- Python versions: `3.10`, `3.11`, `3.12`, `3.13`, `3.14`.
- Architectures: Linux x86_64, Linux aarch64, macOS x86_64, macOS arm64.
- Skipped: musllinux (mojo requires glibc), i686 (mojo is 64-bit only), Windows, PyPy.
- Build tooling uses `uv` + `mojo` for compiling the shared library.
- Linux aarch64 builds use QEMU emulation via `docker/setup-qemu-action`.

## Pre-release evidence collection

Run locally from a clean branch before opening a release:

1. `make install`
2. `make check-release`  
   (runs strict lint + tests)
3. `uv build --sdist`
4. `uv build --wheel`
5. `uv run python -m pip install --force-reinstall ./dist/*.whl`
6. `uv run python -c "import mogemma; print('import ok')"`
7. `make benchmark`
8. `uv run python tools/benchmark.py --mode generation --rounds 30 --max-new-tokens 64 > docs/baseline-generation.json`
9. `uv run python tools/benchmark.py --mode embedding --rounds 30 > docs/baseline-embedding.json`
10. `git status` (ensure only intentional release-related changes are staged)

Capture command output with timestamps in release notes.

## Pre-release matrix checks

1. `uv run pytest src/py/tests`
2. `make build`
3. `uv run ruff check src/py`
4. `uv run ruff format --check src/py`
5. `uv run mypy src/py/mogemma`
6. `uv run pyright src/py/mogemma`
7. `make benchmark`
8. Performance baselines:
   - `docs/baseline-generation.json`
   - `docs/baseline-embedding.json`

## GitHub release workflow (trusted publishing)

1. Push release commit to the target branch.
2. Create and push an annotated tag, e.g. `v0.1.0`.
3. Create a GitHub release from the tag.
4. Confirm `.github/workflows/publish.yml` job runs successfully:
   - `build-source`
   - `build-wheels` (ubuntu-latest, macos-latest)
   - `publish-release`

`publish-release` publishes artifacts from built wheels and source archives after smoke import checks.

## Rollback and remediation

- If runtime or install checks regress, block publish and document the issue before retry.
- If an incorrect build was already published:
  - Do not remove the file.
  - Cut a follow-up release with fixes and explicit package notes.
  - Deprecate affected release in `release` notes as required.

## Release dry-run (optional)

1. Trigger `.github/workflows/publish.yml` by creating a pre-release tag in a scratch branch if your environment supports trusted publishing.
2. Confirm wheel build and test jobs pass end-to-end.
3. Keep generated artifacts in a temporary folder for one day for manual inspection.
4. Capture latest baseline JSON artifacts and include them with release evidence.
5. If publish fails, do not retry immediately with the same commit; patch, rebuild, and re-run with a corrected tag flow.

## Dry-run evidence checklist

- `docs/performance-baselines.md` reflects the same command set and thresholds used in the captured baseline files.
- `docs/baseline-generation.json` and `docs/baseline-embedding.json` are checked into release evidence.

## Go/no-go checklist

- [ ] `make check-release` passes.
- [ ] `python`/`ci` quality jobs pass on the release commit.
- [ ] wheel and sdist artifacts are reproducibly built.
- [ ] smoke import checks succeed from built artifacts.
- [ ] quality, docs, and perf evidence artifacts are attached to release notes.
- [ ] `docs/release-runbook.md` and `docs/performance-baselines.md` reflect current evidence.
