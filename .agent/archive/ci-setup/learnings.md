# Learnings: CI Setup

## [2026-02-15] - GitHub Actions CI pipeline

- **Implemented:** `ci.yml` (lint, type-check, test matrix 3.10-3.13), `pr-title.yml` (conventional commits).
- **Learnings:**
  - Mojo compiler requires custom extra index URL for `uv`.
  - No Mojo compilation/testing in CI — Python tests sufficient.
  - Modeled after sqlspec CI patterns.
  - No slotscheck or pre-commit needed for this project.
