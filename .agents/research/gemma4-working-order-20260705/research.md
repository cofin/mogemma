# Gemma 4 Working Order Research - 2026-07-05

## Executive Summary

This repository is not in working order under the current toolchain. The Python
surface passes its focused tests, but the Mojo runtime does not compile under the
installed Mojo nightly because nullable pointer handling is stale. Gemma 4 support
is also incomplete: the repo models a four-variant Gemma 4 world, while the
official model family now includes five target sizes, including the June 3, 2026
Gemma 4 12B release.

The biggest architectural gap is that Gemma 4 12B is not a simple enum/catalog
addition. Official docs describe it as an encoder-free unified multimodal model
that feeds image patches and audio frames directly into the LLM backbone. The
current codebase is built around separate vision/audio encoder paths, so 12B needs
a dedicated architecture discovery, conversion, prompt/template, and runtime plan.

Recommended repair order:

1. Restore Mojo compilation against the current nightly.
2. Reuse the Gemma 3 architecture harness as prior art: family detection,
   shape-derived runtime parameters, architecture-specific converters, and parity
   gates.
3. Make model support explicit: official model catalog vs downloadable GCS models
   vs runtime-implemented models.
4. Create a dedicated Gemma 4 12B support spec before coding it.
5. Stop advertising or silently accepting unsupported audio paths.
6. Fix mypy/toolchain drift and then clean stale docs, scripts, and Flow state.

## Latest Gemma 4 Model Research

Sources checked on 2026-07-05:

- Google AI Gemma releases: https://ai.google.dev/gemma/docs/releases
- Google AI Gemma 4 overview: https://ai.google.dev/gemma/docs/core
- Google AI Gemma 4 model card: https://ai.google.dev/gemma/docs/core/model_card_4
- Google AI MTP guide: https://ai.google.dev/gemma/docs/mtp/mtp
- Google AI thinking guide: https://ai.google.dev/gemma/docs/capabilities/thinking
- Google Developers Gemma 4 12B guide: https://developers.googleblog.com/gemma-4-12b-the-developer-guide/
- Google DeepMind Gemma 4 page: https://deepmind.google/models/gemma/gemma-4/

Official state as of 2026-07-05:

- 2026-03-31: Gemma 4 released with E2B, E4B, 31B, and 26B-A4B variants.
- 2026-04-16: MTP support released for E2B, E4B, 31B, and 26B-A4B.
- 2026-06-03: Gemma 4 12B Unified released.
- Official target model ids are:
  - `google/gemma-4-E2B-it`
  - `google/gemma-4-E4B-it`
  - `google/gemma-4-12B-it`
  - `google/gemma-4-31B-it`
  - `google/gemma-4-26B-A4B-it`
- Official MTP assistant ids append `-assistant` to each target model id.
- Official docs describe Gemma 4 as supporting native system prompts and thinking
  mode. The thinking chat templates differ by family: E2B/E4B use a different
  marker pattern from 12B/26B-A4B/31B.

Memory guidance from official docs:

| Model | BF16 | SFP8 | Q4_0 | Notes |
| --- | ---: | ---: | ---: | --- |
| E2B | 11.4 GB | 5.7 GB | 2.9 GB | Edge/mobile-oriented |
| E4B | 17.9 GB | 8.9 GB | 4.5 GB | Edge/mobile-oriented |
| 12B | 26.7 GB | 13.4 GB | 6.7 GB | Unified multimodal 12B |
| 26B-A4B | 57.7 GB | 28.8 GB | 14.4 GB | MoE, 4B active |
| 31B | 69.9 GB | 34.9 GB | 17.5 GB | Dense workstation class |

Impact on Mogemma:

- The repo is missing 12B from its architecture enum, docs, GCS/default catalog,
  conversion assumptions, and runtime pathways.
- 12B support must account for encoder-free image/audio handling, not just the
  existing separate `vision_config` and audio encoder assumptions.
- MTP and thinking mode should be modeled explicitly. They affect model ids,
  prompt formatting, and likely generation API surface.

## Gemma 3 Prior Art

The earlier Gemma 3 implementation is important local evidence. It was not just a
stale documentation branch; it shows the structure this repo used when the
runtime could support multiple Gemma architectures.

Reference points:

- `v0.3.0:src/py/mogemma/model.py`
- `v0.3.0:src/py/mogemma/convert.py`
- `v0.3.0:src/mo/mogemma/core.mojo`
- `v0.3.0:src/mo/mogemma/model.mojo`
- `v0.3.0:src/mo/mogemma/layers.mojo`
- `.agents/archive/model-architecture-support/prd.md`
- `.agents/archive/gemma3-standard-hardening/spec.md`
- `.agents/archive/gemma3-nano-conversion/spec.md`
- `.agents/archive/gemma3-nano-inference/spec.md`

What worked in the Gemma 3 design:

- Architecture detection happened early. Mojo detected Nano from
  `model.layers.0.altup.router.weight`; Python conversion detected Nano from
  `altup` or `per_layer_mapping` tensor names.
- Standard and Nano had separate runtime structs and forward paths instead of
  one generic path with implicit assumptions.
- Runtime geometry was derived from tensors. For example, `head_dim` came from
  `q_norm.shape_0`, and embedding output width was checked against the model's
  hidden size rather than a fixed 768.
- Architecture overrides existed as an explicit bridge from Python config into
  Mojo init.
- Nano conversion was a family-specific mapper with validation of per-layer
  embeddings, AltUp coefficients, Laurel tensors, and per-layer mapping shapes.
- Unsupported or ambiguous shapes were meant to fail at init/conversion with
  concrete messages rather than silently producing wrong results.
- The test suite covered core runtime init, Nano stepping, Nano embeddings,
  descriptor reuse, KV-share behavior, conversion shape checks, and CPU/GPU
  parity scaffolding.

The Gemma 4 rewrite/regression pattern:

- `git diff --stat v0.3.0..v0.4.0` shows a large replacement, not a small
  extension: about 10k insertions and 10k deletions across runtime and tests.
- The transition deleted `src/py/tests/test_mojo_core.py`,
  `src/py/tests/test_convert.py`, `src/py/tests/test_nano_parity_gates.py`, and
  `src/mo/tests/test_nano_layers.mojo`.
- Gemma 4 tests were added, but the old architecture-dispatch and parity harness
  was not carried forward at the same strength.
- The current Mojo pointer failure is partly a compiler API drift issue, but the
  deeper working-order gap is that the Gemma 3-style architecture contract was
  weakened during the Gemma 4 cutover.

Implication for Gemma 4:

- Do not treat Gemma 3 only as stale content to delete. Mine it for the support
  architecture.
- Port the pattern, not the model semantics. Gemma 4 needs its own variant
  detector, converter validators, runtime structs, and parity gates, but the
  Gemma 3 scaffolding is the right template.
- The 12B implementation should follow the old Nano precedent: recognize the
  architecture first, validate tensor layout, route to a dedicated path, and add
  explicit unsupported-mode errors until parity evidence exists.

## Codebase Analysis

### Python API and model metadata

Relevant files:

- `src/py/mogemma/model.py`
- `src/py/mogemma/hub.py`
- `src/py/mogemma/convert.py`
- `src/py/mogemma/orbax_loader.py`
- `src/py/mogemma/backends.py`

Findings:

- `Gemma4Variant` currently contains E2B, E4B, 31B, and 26B-A4B only. There is
  no 12B variant.
- `_detect_gemma4_variant()` defaults non-PLE/non-MoE configs to 31B. A 12B
  config would likely be misclassified as 31B unless explicit detection is added.
- `KNOWN_GCS_MODELS` currently lists only E2B-it, E4B-it, and 26B-A4B-it. That
  may be correct for the current public GCS bucket, but it is not an official
  Gemma 4 support matrix.
- The repo needs separate concepts:
  - official Gemma 4 model ids
  - models available from the configured GCS source
  - models implemented and validated by Mogemma
  - models known but intentionally unsupported
- `_format_gemma4_prompt()` still uses the older `<start_of_turn>` style. July
  2026 Gemma 4 thinking docs show newer template forms using `<|turn>`,
  `<|think|>`, and `<|channel>thought` depending on model family and thinking
  mode.
- `OrbaxLoader` eagerly materializes every tensor. That is not viable as a
  runtime path for 26B, 31B, or 12B-scale checkpoints.
- GCS download code materializes each remote object into bytes before writing it
  to disk. This is risky for large checkpoint shards.
- `_write_sharded()` is described as streaming, but it buffers whole shard
  dictionaries before writing. For multi-GB models this can exceed practical
  memory limits.
- `_generate_config_json()` assumes layer 0 has `attn.kv_einsum.w`. That is too
  narrow for future 12B layout work and any config with separate k/v tensors.
- The Python GPU availability layer is environment-variable gated
  (`MOGEMMA_GPU_AVAILABLE`) while the Mojo runtime has real accelerator probing.
  This is useful for tests but poor as the user-facing device selection default.

### Mojo runtime

Relevant files:

- `src/mo/mogemma/model.mojo`
- `src/mo/mogemma/core.mojo`
- `src/mo/mogemma/gpu_context.mojo`
- `src/mo/mogemma/gpu_pipeline.mojo`
- `src/mo/mogemma/weight_stage.mojo`

Findings:

- Current Mojo tests fail because the code constructs null
  `UnsafePointer[..., MutExternalOrigin]` values. The current Mojo nightly treats
  `UnsafePointer` as non-nullable and requires `Optional[UnsafePointer]` for
  nullability.
- The runtime also emits many origin deprecation warnings. `MutExternalOrigin`
  should be migrated to current origin conventions or explicit origin casts.
- Audio support is not wired end-to-end. `core.mojo` has pass-only hydration
  blocks and can produce zero audio embeddings instead of failing clearly when
  weights are missing or unsupported.
- A stale comment says MoE streaming is a placeholder even though
  `forward_gemma4_moe_step()` is called. Comments now make the code harder to
  trust.
- GPU resource planning appears dense-weight oriented and should be rechecked
  against MoE, vision, and future 12B unified multimodal paths.
- Weight staging uses scalar copy loops from mapped weights into host buffers.
  That may be acceptable for correctness first, but it should be measured before
  performance claims.

### Documentation and Flow state

Relevant files:

- `README.md`
- `docs/architecture/device-selection.md`
- `docs/architecture/concepts/how-it-works.md`
- `docs/architecture/concepts/diagrams.md`
- `docs/architecture/concepts/standard-vs-nano.md`
- `.agents/product.md`
- `.agents/tech-stack.md`
- `.agents/flows.md`
- `.agents/specs/*/spec.md`

Findings:

- README model selection lists only E2B, E4B, and 26B-A4B. It omits 31B and the
  newly official 12B.
- README examples still use `google/gemma-4-E4B` without `-it`, while the current
  GCS catalog only lists `google/gemma-4-E4B-it`.
- Several architecture docs still describe Gemma 3/Nano paths as active even
  though the current project docs say the repo has cut over to Gemma 4.
- `.agents/product.md` says Gemma 3 support is legacy. `.agents/tech-stack.md`
  says Gemma 3 is not supported and no Gemma 3 code exists. These cannot both be
  the canonical policy.
- `.agents/flows.md` still references an archived default model
  `google/gemma-4-E4B`, while code now defaults to an instruction-tuned model.
- Active specs are stale. Some tasks remain unchecked even though code appears
  implemented; some comments point at old knowledge file names.

## Dead, Stale, and Bad Code

### Dead or probably obsolete

- `scripts/dump_orbax_inventory.py`: looks like a phase-1 probe script that the
  conversion spec expected to delete after inventory work. Either archive it as a
  runbook tool or remove it.
- Old Gemma 3/Nano architecture docs should not be presented as live behavior,
  but they should be preserved or distilled as prior-art notes for the Gemma 4
  architecture harness.
- Stale Flow archive/default references to non-IT Gemma 4 model ids.

### Misleading or stale comments

- `src/py/mogemma/convert.py` still documents `NotImplementedError` for MoE
  conversion, but the code now contains a MoE conversion path.
- `src/mo/mogemma/core.mojo` still describes MoE streaming as a placeholder.
- The converter references an old knowledge path name in comments/spec language.

### Bad behavior or unsafe defaults

- Mojo nullable pointer model is incompatible with the current compiler.
- Audio input can be accepted without real audio execution, leading to silent
  zero embeddings or incomplete hydration.
- Raw Orbax loading is eager and unsafe for large models.
- Download and conversion paths buffer too much data for large checkpoints.
- Model support is implied by docs/config in places where there is no validation
  evidence.

### Needs restructuring

- Split model registry into structured support layers:
  - `OfficialModel`
  - `DistributionAvailability`
  - `RuntimeSupport`
  - `ValidationStatus`
- Split architecture parsing from runtime support checks. A model can be
  recognized but rejected with a precise unsupported-feature error.
- Split multimodal support by architecture:
  - separate vision/audio encoder family
  - 12B unified encoder-free family
- Separate prompt templating by model family and thinking mode instead of one
  hardcoded Gemma 4 template.
- Make conversion a true streaming pipeline, with low memory shard emission and
  no all-tensor runtime fallback for large Orbax checkpoints.
- Define a single device availability contract that bridges Python API choices
  and Mojo accelerator probing.

## Risk Assessment

High-risk areas:

- 12B implementation: architecture differs from current runtime assumptions.
- Audio: current public API is ahead of implementation.
- Pointer migration: broad Mojo type changes can touch most model/weight structs.
- Converter changes: memory fixes can alter file naming, tensor order, and shard
  layout.

Medium-risk areas:

- Prompt template updates: correctness depends on tokenizer/chat template
  compatibility.
- Model registry split: public API docs and default selection behavior may need
  compatibility aliases.
- GCS vs official model catalog: users may expect official ids to download even
  when the configured bucket does not host them.

Low-risk areas:

- Documentation cleanup.
- Removing or archiving one-off scripts after confirming no automation depends on
  them.
- Stale comment/spec synchronization.

## Recommended Flow Sequence

### Flow 1: Mojo nightly repair

Goal: make Mojo tests compile and pass on the current Mojo nightly.

Tasks:

- Replace null `UnsafePointer` fields with `Optional[UnsafePointer]` or raw
  address fields with explicit nonzero conversion helpers.
- Migrate deprecated pointer origins.
- Keep changes mechanical first; do not restructure model logic in the same pass.
- Validate with `uv run pytest -q src/mo/tests/test_mojo.py`.

### Flow 2: Model registry and support matrix

Goal: restore the Gemma 3-style support contract before adding 12B.

Tasks:

- Reintroduce architecture-dispatch tests analogous to the v0.3 Gemma 3
  `test_mojo_core.py`, `test_convert.py`, and Nano parity gates, adapted for
  Gemma 4.
- Add an official Gemma 4 catalog containing E2B, E4B, 12B, 31B, and 26B-A4B.
- Keep `KNOWN_GCS_MODELS` or equivalent as source-specific availability, not
  official support.
- Add runtime support statuses for text, vision, audio, MoE, MTP, thinking, and
  quantization.
- Update README and API errors to report exact unsupported reasons.

### Flow 3: Gemma 4 12B architecture spike

Goal: create a small, evidence-backed design before implementation.

Tasks:

- Capture actual 12B config and tensor names from the chosen distribution source.
- Define parser behavior for unified encoder-free multimodal config, following
  the Gemma 3 Nano precedent: detect by tensor/config features, validate layout,
  then route to a dedicated architecture path.
- Decide whether 12B first target is text-only, image+text, audio+text, or full
  multimodal.
- Add tests proving 12B is recognized and either routed or rejected precisely.

### Flow 4: Audio truth fix

Goal: remove false audio support.

Tasks:

- If audio weights are absent or unsupported, raise explicit unsupported-feature
  errors.
- If audio is implemented, wire hydration and forward paths with parity tests.
- Update docs to say exactly which models and modes support audio.

### Flow 5: Converter and download memory hardening

Goal: make large model preparation realistic.

Tasks:

- Stream GCS object downloads to disk.
- Stream safetensors shard writing without retaining all shard arrays.
- Remove or gate eager raw Orbax runtime loading for large models.
- Add memory-budgeted tests or synthetic large-tensor regression tests.

### Flow 6: Docs, specs, and cleanup

Goal: align public docs, Flow state, and code.

Tasks:

- Remove Gemma 3/Nano claims from live product docs, but preserve the useful
  architecture-dispatch lessons as prior-art documentation.
- Fix README model table and examples.
- Synchronize `.agents/product.md`, `.agents/tech-stack.md`, and `.agents/flows.md`.
- Close or refresh stale spec checkboxes.
- Archive or remove obsolete probe scripts.

## Validation Evidence

Commands run during this research pass:

- `bd ready`
  - Result: no open issues.
- `uv run ruff check src/py`
  - Result: passed.
- `uv run ruff format --check src/py`
  - Result: passed.
- `uv run pyright`
  - Result: 0 errors.
- `uv run mypy`
  - Result: failed before project checking because installed NumPy stubs use
    Python 3.12 `type` statement syntax while mypy is configured for Python 3.10.
- `uv run pytest -q src/py/tests`
  - Result: 295 passed, 4 skipped.
- `uv run pytest -q src/mo/tests/test_mojo.py`
  - Result: 13 failed, 2 passed, 1 skipped.
  - Dominant failure: current Mojo rejects null construction of non-nullable
    `UnsafePointer`; use `Optional[UnsafePointer]` for nullable pointers.
- `git diff --stat v0.3.0..v0.4.0 -- src/py/mogemma src/mo/mogemma src/py/tests src/mo/tests .agents`
  - Result: about 10k insertions and 10k deletions; Gemma 4 was a broad rewrite.
- `git diff --name-status v0.3.0..v0.4.0 -- ...`
  - Result: deleted the old core runtime/converter/parity test harnesses while
    adding Gemma 4-specific tests.

## Bottom Line

The repo should not start by "adding Gemma 4 12B" directly. First make the Mojo
runtime compile, then restore the Gemma 3-style architecture contract: detect the
model family, validate tensor layout, route to dedicated runtime paths, and keep
parity tests around those paths. After that, split the model catalog from actual
runtime support and add 12B behind a dedicated architecture plan. That gives the
project a truthful support matrix and avoids repeating the current pattern where
docs, defaults, converters, and runtime capabilities drift independently.
