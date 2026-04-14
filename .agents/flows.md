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
*In progress. Base transformer / PLE / Vision iterators + sharded writer + config.json synthesis shipped. Hub wiring (3.1/3.3) + MoE iterator (2.2) + integration tests (4.x) pending.*
*Link: [./specs/orbax-safetensors-conversion/spec.md](./specs/orbax-safetensors-conversion/spec.md)*
