# mogemma knowledge base

Curated, component-oriented reference for how mogemma works. Each file is a
living document — update it in place when the underlying code changes. Do NOT
add per-flow or per-spec "completion" files here; those belong in
`.agents/specs/<flow-id>/learnings.md` if anywhere.

## Contents

| File | Covers |
|---|---|
| [architecture.md](architecture.md) | Top-level system design: Python ↔ Mojo bridge, package layout, module boundaries |
| [gemma4-models.md](gemma4-models.md) | Gemma 4 variants (E2B, E4B, 31B, 26B-A4B), PLE, MoE two-branch design, vision/audio encoders, attention patterns |
| [python-runtime.md](python-runtime.md) | `src/py/mogemma/`: hub, orbax_loader, safetensors_loader, convert, model, engine, tokenizer, config |
| [mojo-runtime.md](mojo-runtime.md) | `src/mo/mogemma/`: core/layers/model/gpu_context, struct layouts, weight staging, FFI entry points |
| [gpu-infrastructure.md](gpu-infrastructure.md) | Backend trait, kernel set, KV cache, weight staging tiers, device selection |
| [build-and-packaging.md](build-and-packaging.md) | hatchling + hatch-mojo, cibuildwheel, manylinux + libstdcxx quirks, wheel repair, PyPI |
| [ci-and-release.md](ci-and-release.md) | GitHub workflows, release flow, rust-downloader FFI, obstore |
| [quality-gates.md](quality-gates.md) | make lint/test/check, zero-warning policy, ruff/mypy/pyright config, flaky tests |
| [performance.md](performance.md) | Benchmark harness, baselines, reproducibility |
| [learnings.md](learnings.md) | Hard-won gotchas that don't fit a single component |

## How to use

- **Looking up "how does X work?"** → start with `architecture.md` for orientation, then the component file.
- **Adding new knowledge** → find the right file and edit it. Do not create a new file per flow. If truly cross-cutting, add to `learnings.md`.
- **Found a stale fact?** → fix it in place; git history preserves the old version.

## Related

- `.agents/code-styleguides/python.md` — Python house style
- `.agents/code-styleguides/mojo.md` — Mojo house style
- `.agents/code-styleguides/testing.md` — test conventions
- `.agents/code-styleguides/bash.md` — shell scripting conventions
- `.agents/tech-stack.md` — approved technologies and versions
- `.agents/workflow.md` — Flow process and Beads integration
