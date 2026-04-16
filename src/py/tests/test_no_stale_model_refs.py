"""Repo-wide grep gate: no ``gemma3-*`` or ``gemma3n-*`` references in live code.

Gemma 3 and Gemma 3n were retired when the project migrated to Gemma 4 in
commit ``0f5e349``. A live probe of ``gs://gemma-data`` on 2026-04-16 confirms
only Gemma 4 checkpoints are published — any residual reference to an older
generation is a latent failure.

Scope intentionally excludes:
- Test files (they legitimately embed the forbidden strings as assertion
  payloads and documentation).
- ``.agents/`` (historical specs, archived flows, learnings — kept for
  project memory and explicitly allowed to mention retired IDs).

Any new false positive should be addressed by fixing the live code path,
not by widening the skip list.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

_REPO_ROOT = Path(__file__).resolve().parents[3]

_SCAN_ROOTS = (
    _REPO_ROOT / "src" / "py" / "mogemma",
    _REPO_ROOT / "src" / "mo",
    _REPO_ROOT / "tools",
    _REPO_ROOT / ".github",
)

_STALE_PATTERN = re.compile(r"gemma3[n-]|gemma3-", re.IGNORECASE)

# File-name prefixes that identify test files; they legitimately reference
# retired model IDs in assertions / docstrings that verify absence.
_TEST_PREFIXES = ("test_",)


def _iter_source_files() -> Iterator[Path]:
    for root in _SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in {".py", ".mojo", ".yml", ".yaml", ".md", ".toml"}:
                continue
            if any(path.name.startswith(prefix) for prefix in _TEST_PREFIXES):
                continue
            yield path


def test_no_live_gemma3_or_gemma3n_references() -> None:
    offenders: list[str] = []
    for path in _iter_source_files():
        text = path.read_text(errors="replace")
        for match in _STALE_PATTERN.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            rel = path.relative_to(_REPO_ROOT)
            offenders.append(f"{rel}:{line_no}: {match.group(0)!r}")

    assert not offenders, "Stale Gemma 3/3n references in live code:\n" + "\n".join(offenders)
