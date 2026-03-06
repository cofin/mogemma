# Project Patterns

> Consolidated learnings and patterns from all flows.
> This file is the single source of truth for project conventions.

## Code Conventions

- **Mojo Style:** Use `snake_case` for variables and functions.
- **Python Typing:** Always use `>=3.10` types (built-in types like `list`, `dict`).
- **Docstrings:** Required for all public functions/methods.

## Architecture Patterns

- **Hybrid Mojo-Python:** Use `hatch-mojo` plugin (configured in `pyproject.toml` under `[[tool.hatch.build.targets.wheel.hooks.mojo.jobs]]`) for Mojo compilation integration.
- **Contract Testing:** Use `test_altup_contract.mojo` style for testing cross-language boundaries.
- **Packaging Baseline:** Use `hatch-mojo>=0.1.5` with `bundle-libs = true` as canonical wheel policy; only keep Linux-specific overrides when they have explicit rationale and exit criteria.

## Gemma Model Patterns

- **Instruction-Tuned (`*-it`) Models:** Require turn formatting (`<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n`) for reliable output. Raw prompts produce low-quality artifacts.
- **EOS Handling:** Prioritize `<end_of_turn>` as EOS for `*-it` models; otherwise generation drifts into repeated turn markers.
- **Deterministic Validation:** Use `temperature=0, top_k=1` for factual contract checks; stochastic decoding masks regressions.

## Gemma3 Nano-Specific Patterns

- **KV Sharing:** Layers 20..29 in `gemma3n-e2b-it` have zeroed `k_proj`/`v_proj`; KV-sharing must be implemented (not normal attention).
- **Weighted RMSNorm:** Nano norm tensors use `w * x` (without `1 + w`), plus `v_norm` on values, and attention scaling of `1.0`.
- **Activation Sparsity:** Early decoder layers (first 10) depend on Gaussian top-k gating (default `0.95`); skipping degrades quality.
- **AltUp Dimensions:** Per-layer dimensions are authoritative from `per_layer_map.gate.weight`; inferring `per_layer_dim` from `per_layer_embed.shape_1` is unsafe with 3D embeddings.
- **Conversion Validation:** Conversion-shape correctness is necessary but insufficient; runtime must faithfully implement AltUp + per-layer-input + KV-sharing semantics.

## Gotchas & Warnings

- **Mojo Pathing:** Ensure `MOJO_PATH` includes `src/mo`.
- **Hatch/UV:** Use `uv run` to ensure virtualenv consistency.
- **Model Rebuild Overhead:** Cache runtime descriptors at init; hot-path model rebuild per-call degrades performance.

## Testing Patterns

- **Pytest-Mojo:** Use `pytest` to drive Mojo tests via the standard CLI.
- **Contract Tests:** Cross-language boundaries tested via `test_altup_contract.mojo` style.

## Context for AI Assistants

- **Source Layout:** Mojo code is in `src/mo/mogemma`, Python code in `src/py/mogemma`.
