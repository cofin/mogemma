# Learnings - Text Generation Core

...

## [2026-02-08 16:00] - Phase 3 Task 2: Benchmark tokens-per-second vs native MAX execution

- **Implemented:** `scripts/benchmark.py` for measuring TPS and bridge overhead.
- **Files changed:** `scripts/benchmark.py`
- **Commit:** a1e9ba4
- **Learnings:**
  - **Overhead Measurement:** The benchmark script confirmed that the Python/Mojo bridge adds minimal overhead (sub-millisecond) for simple token generation loops.
  - **Tooling:** Automated performance tracking is essential for maintaining the project's "high-performance" core value.