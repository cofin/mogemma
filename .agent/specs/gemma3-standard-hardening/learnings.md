# Learnings: gemma3-standard-hardening

## 2026-03-05

- Architecture overrides should be validated in config objects (`dict[str, int|float]`) before model init, otherwise errors surface too late and are harder to diagnose.
- Model-family detection is more reliable from tensor metadata keys than from user-provided model ids; nano-specific keys such as `.per_layer_map.` and `.laurel.` allow deterministic variant gating.
- Passing architecture overrides directly to `_core.init_model` must be explicit and fail-fast when the extension signature is incompatible, to avoid silently ignoring runtime-shaping inputs.
