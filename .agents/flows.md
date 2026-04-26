# Flow Registry

This file tracks all flows (features, bugs, refactors) for the project.
Each flow has its own detailed spec and plan in its respective folder.

---

## [x] Flow: HF URL Fix (Epic: mogemma-40q.1)
*Link: [./archive/hf-model-alignment_20260413/hf-model-alignment/hf-url-fix/spec.md](./archive/hf-model-alignment_20260413/hf-model-alignment/hf-url-fix/spec.md)*

## [x] Flow: Default Model Update (Epic: mogemma-40q.2)
*Link: [./archive/hf-model-alignment_20260413/hf-model-alignment/default-model-update/spec.md](./archive/hf-model-alignment_20260413/hf-model-alignment/default-model-update/spec.md)*

## [x] Flow: GCS Download Backend (Chapter 1)
*Archived 2026-04-14. Shipped via `a59de41` (hub → GCS migration).*
*Link: [./archive/gcs-download-backend_20260414/spec.md](./archive/gcs-download-backend_20260414/spec.md)*

## [x] Flow: HF Removal & Cleanup (Chapter 4)
*Archived 2026-04-14. Verified: zero HF imports/URLs remain in `src/py/mogemma/`.*
*Link: [./archive/hf-removal-cleanup_20260414/spec.md](./archive/hf-removal-cleanup_20260414/spec.md)*

## [x] Flow: PT Embedding Auto-select (Chapter 3)
*Archived 2026-04-14. Shipped via `a511399` — `EmbeddingConfig` defaults to `google/gemma-4-E4B` pretrained.*
*Link: [./archive/pt-embedding-autoselect_20260414/spec.md](./archive/pt-embedding-autoselect_20260414/spec.md)*

---

## PRD: GCS Download + Orbax→Safetensors Migration
*Parent PRD for the chapters above. Active while orbax-safetensors-conversion finishes the MoE path.*
*Link: [./specs/gcs-orbax-migration/prd.md](./specs/gcs-orbax-migration/prd.md)*

## [~] Flow: Orbax→Safetensors Conversion (Chapter 2)
*In progress. Base / PLE / Vision / MoE Python iterators + sharded writer + config.json synthesis + hub integration + round-trip test all shipped. Remaining: Mojo-side MoE runtime (split into sub-flow) and manual 26B verification.*
*Link: [./specs/orbax-safetensors-conversion/spec.md](./specs/orbax-safetensors-conversion/spec.md)*

## [ ] Flow: MoE Mojo Runtime (Chapter 2b)
*Planning. Mojo-side rewrite to consume the two-branch MoE safetensors contract: struct layout, hydration, GPU packer, router + packed-expert kernels, two-branch forward pass. Blocks live 26B-A4B-it inference.*
*Link: [./specs/moe-mojo-runtime/spec.md](./specs/moe-mojo-runtime/spec.md)*

## [~] Flow: Fix Stale Defaults
*Active. `EmbeddingConfig` default + every `tools/*.py` + `check-release` CI step reference models that are missing from `gs://gemma-data` after the hub migration. Unblocks package defaults and the release runbook. Landing directly on `fix/timeout` (PR #16) per user direction.*
*Link: [./specs/fix-stale-defaults/spec.md](./specs/fix-stale-defaults/spec.md)*

## [ ] Flow: GPU CI on Modal (Tier C)
*Queued. Stand up Modal-based GPU runner for `test_gpu_*.mojo` on every PR + on-demand real-weight parity gates. Follow-on to `fix-stale-defaults`.*

## [ ] Flow: Mojo Tests in CI
*Queued. Add `src/mo/tests` to the CI pytest target so CPU-path Mojo regressions are caught even without GPU hardware. Free, high-leverage, no storage required.*

## [ ] Flow: Metal DeviceContext Workaround
*Queued. Investigate the stdlib `metal:1-metal4` arch-list rejection that forces `has_usable_gpu()` to skip Metal entirely. Unblocks GPU coverage on M-series developer hardware.*

## [ ] Flow: Vision GPU Null-Ptr Fix
*Queued. Fix the vision-encoder null-pointer rebind in the GPU upload path that was gated out in `657f8ed`. Prerequisite for re-enabling vision on GPU.*

## [~] Flow: Gemma Softcapping
*Link: [./specs/gemma-softcapping/](./specs/gemma-softcapping/)*
