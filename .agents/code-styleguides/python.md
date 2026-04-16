# Python Style Guide

House style for Python code in `src/py/mogemma/` and `src/py/tests/`. Tooling
enforces most of this; the rest is convention. See
`.agents/knowledge/quality-gates.md` for gate details.

## Tooling

- **Formatter + linter**: Ruff (`ruff check` + `ruff format`), preview features on.
- **Type checkers**: mypy AND pyright — both must pass. Don't rely on one catching what the other misses.
- **Docstrings**: Google style (`[tool.ruff.lint.pydocstyle] convention = "google"`).
- **Line length**: 120. Docstring code blocks: 60.
- **Python version**: 3.10 target, tested 3.10–3.13.

Run `make lint` before every commit. CI runs the same command.

## File structure

```python
"""Module docstring — one-line summary, then details if needed."""

from __future__ import annotations      # always, enables PEP 604 on 3.10

import asyncio                            # stdlib first, alphabetical
import logging
from pathlib import Path                  # from-imports grouped with the section
from typing import Any                    # only when unavoidable

import numpy as np                        # third-party, alphabetical
import obstore as obs

from mogemma.config import GenerationConfig   # first-party last

logger = logging.getLogger(__name__)        # module logger immediately after imports

_MODULE_CONSTANT = 42                        # private constants near the top
```

## Typing

- **Full annotations on public API.** Every function signature and return type.
- Prefer `X | None` over `Optional[X]` (PEP 604; `from __future__ import annotations` covers 3.10).
- Prefer `list[X]`, `dict[K, V]` over `List`/`Dict`.
- `Any` is an escape hatch, not a default — justify with a comment when used.
- Use `TYPE_CHECKING` for imports needed only by type checkers (Ruff auto-fixes this via `TC` rules).
- Generic internal helpers can drop param annotations; public API cannot.

## Naming

- `snake_case` for functions and variables.
- `PascalCase` for classes.
- `SCREAMING_SNAKE_CASE` for module constants.
- Leading underscore for module-private (`_helper`).
- Leading underscore for instance-private (`self._cache_path`).
- No double-underscore name-mangling unless you actually want it.

## Error handling

- **No silent fallbacks.** If resolution/download/parse fails, raise. Log the miss, then raise.
- **Explicit error taxonomy.** Custom exceptions (`ModelNotFoundError`) for user-facing failure modes. Don't collapse everything into `RuntimeError`.
- **Never bare `except:`.** Catch the narrowest exception that makes sense. Re-raise if you can't actually handle it.
- **Re-raise with context**: `raise FooError(...) from original_exc`.
- Don't use exceptions for control flow in hot paths.

## Logging

- Use the module logger (`logger = logging.getLogger(__name__)`).
- Levels:
  - `logger.debug` — developer-only diagnostics.
  - `logger.info` — user-visible milestones.
  - `logger.warning` — recoverable anomaly worth seeing.
  - `logger.error` — failure path; always include exception info via `exc_info=True` or `logger.exception`.
- No `print` in library code. `print` in scripts/CLI is OK.

## Async

- `anyio` primitives over raw `asyncio` — the test suite runs under trio too.
- `async def` functions do async work; if they don't, don't mark them async.
- Never swallow `CancelledError`.
- Use `anyio.CancelScope` over signal-based cancellation.
- Backpressure: use `Semaphore` for concurrency limits (e.g., `_ASYNC_DOWNLOAD_CONCURRENCY = 6`).

## Paths

- Use `pathlib.Path` everywhere. No raw string paths in API surface.
- `str(path)` only at FFI boundaries or when a third-party lib demands it.

## Classes

- Dataclasses / pydantic for pure-data types.
- `__init__` stays short; heavy construction goes in a classmethod factory (`HubManager.from_config(...)`).
- No `@property` for trivial attribute access — use the attribute directly.
- `__slots__` only where memory matters (hot loops, large fleets of instances).

## Docstrings

Google style. Required on public API; optional on private helpers if the
signature makes intent obvious.

```python
def download_sync(self, model_id: str) -> Path:
    """Download a model from GCS and return the local cache path.

    Args:
        model_id: Hugging Face-style id, e.g. ``google/gemma-4-e2b-it``.

    Returns:
        Absolute path to the finalized local directory.

    Raises:
        ModelNotFoundError: If the model does not exist in GCS.
    """
```

Do NOT write multi-paragraph docstrings. Summarize in one line, add Args/
Returns/Raises if there's anything non-obvious.

## Comments

- Default to **no comments**. Code + good names should speak.
- Write a comment for the **WHY**, never for the WHAT. If the name/code explains it, delete the comment.
- Never reference PR numbers, current tasks, or issue IDs in comments — that belongs in commit messages.

## Type-narrowing helpers

Prefer explicit `assert isinstance(x, X)` over clever type guards in readable
paths — mypy/pyright both honor it and failures are precise.

## Imports

- `from __future__ import annotations` at the top of every `.py`.
- Group order: `__future__` → stdlib → third-party → first-party.
- Within a group, sort alphabetically (Ruff does this).
- Prefer absolute imports over relative. Relative OK for intra-package utilities.

## Tests

See `.agents/code-styleguides/testing.md` for full testing conventions. mogemma-specific:

- Test files: `src/py/tests/test_*.py`.
- pytest, function-based (not class-based).
- Relaxed rules for tests: `ANN202`, `ARG001`, `S101`, `PLR2004` disabled per `[tool.ruff.lint.per-file-ignores]`.
- Async tests use `@pytest.mark.anyio`.
- Use real fixtures, not default-return stubs, for parity testing.

## Patterns specific to mogemma

### FFI pointer extraction

```python
def _tensor_ptr(arr: np.ndarray) -> int:
    """Return the raw pointer to a C-contiguous numpy array.

    Caller must keep ``arr`` alive for the lifetime of any Mojo-held reference.
    """
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr.ctypes.data
```

Never hand a non-contiguous array's pointer to Mojo. Never let the source
array be GC'd while Mojo still points at it.

### Opaque handles from Mojo

`llm` dict pattern:

```python
llm["_gpu_kv_cache"] = int(gpu_ptr)   # opaque — never dereference in Python
```

### Streaming patterns

When working with multi-GB checkpoints, use streaming helpers
(`OrbaxLoader.enumerate_tensors` / `open_tensor`) — never eager-load.

### Stub pattern for streaming access

```python
stub = OrbaxLoader.__new__(OrbaxLoader)     # bypass __init__
stub.model_path = Path(model_path)
names = stub._enumerate_tensor_names()       # noqa: SLF001  # intentional private access
```

## Anti-patterns (reject on review)

- `Any` in public API without justification comment.
- Bare `except:` or `except Exception:` without re-raise/log.
- Silent fallback ladders without logging.
- `# type: ignore` without a specific error code.
- Comments explaining WHAT the code does.
- Hard-coded absolute paths.
- `print` in library code.
- `from module import *`.
- Lambdas assigned to a name (`f = lambda x: ...`) — use `def`.
- String-concatenated SQL / shell commands — use parameterization.
