# Learnings: Advanced Features & Distribution

## Patterns
- **Async Bridging:** Used `asyncio.to_thread` to wrap blocking synchronous Mojo calls. This allows `mogemma` to be used in high-concurrency async frameworks like FastAPI without blocking the event loop.
- **Local-First Hub:** Implemented `HubManager` which prioritizes local paths, then checks a dedicated cache (`~/.cache/mogemma`), and finally falls back to Hugging Face Hub.
- **Unified CLI:** Used `rich-click` to provide a polished terminal experience for model management and chat.

## Gotchas
- **Packaging Subfolders:** When using `src/py` as a package root, `hatchling` and `uv` need explicit configuration in `pyproject.toml`.
- **Mocking native calls:** In tests, ensure mocks return the exact types (e.g., NumPy arrays) expected by the code, as native extensions often have strict type requirements.
