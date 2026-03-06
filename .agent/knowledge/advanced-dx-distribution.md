# Knowledge: Advanced Features & Distribution

> Flow: advanced-dx-distribution | Archived

## Patterns

- **Async Bridging:** `asyncio.to_thread` wraps blocking Mojo calls for use in async frameworks (FastAPI, etc.) without blocking the event loop.
- **Local-First Hub:** `HubManager` prioritizes local paths → dedicated cache (`~/.cache/mogemma`) → Hugging Face Hub fallback.
- **Unified CLI:** `rich-click` provides polished terminal UX for model management and chat.

## Gotchas

- **Packaging `src/py` subfolders:** `hatchling` and `uv` require explicit configuration in `pyproject.toml` when using `src/py` as package root.
- **Mocking native calls:** Mocks must return exact types (e.g., NumPy arrays) expected by code — native extensions enforce strict type requirements.
