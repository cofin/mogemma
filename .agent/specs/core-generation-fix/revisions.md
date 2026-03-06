## [2026-02-24 00:00] Revision 1

**Type:** both
**Reason:** The LLM generation was producing gibberish ("rowingITUDE") and the user requested that the nano model also be validated.

### Changes Made

**Spec Changes:**
- Added requirement to prepend the BOS token explicitly.
- Added requirement to include NANO_MODEL_ID in `tools/validate.py`.

**Plan Changes:**
- Added: Task 1 - Fix Tokenization & BOS Token
- Added: Task 2 - Validate Nano Model
- Added: Task 3 - Integration Testing

### Impact Assessment

- Tasks affected: mogemma-yj1.2.1, mogemma-yj1.2.2, mogemma-yj1.2.3
- Timeline impact: Immediate implementation required
- Dependencies updated: None

## [2026-02-24 00:05] Revision 2

**Type:** plan
**Reason:** The user requested that the validation script test the gemma embedding-only model.

### Changes Made

**Plan Changes:**
- Modified: Task 2 (mogemma-yj1.2.2) - Updated description to include testing the Gemma text-embedding model (`gemma3-text-embedding-4m`) in the validation script.
- Modified: `spec.md` Phase 2 to include requirement 2.2 (text-embedding test).

### Impact Assessment

- Tasks affected: mogemma-yj1.2.2
- Timeline impact: Negligible
- Dependencies updated: None

## [2026-02-24 18:26] Revision 3

**Type:** both
**Reason:** Updated roadmap priorities and clarified that BOS prepend already exists; primary risk moved to nano-specific Mojo math/tensor mapping plus deterministic semantic quality gates.

### Changes Made

**Spec Changes:**
- Reframed scope to prioritize `gemma3n-e2b-it` gibberish root cause in Mojo nano path.
- Added explicit requirements for deterministic semantic prompt suite (3-5 prompts) for standard model.
- Added strict fail-fast contract language (no soft fallbacks).

**Plan Changes:**
- Updated Task 1 (`mogemma-yj1.2.1`) to focus on nano core defect isolation.
- Updated Task 2 (`mogemma-yj1.2.2`) to focus on standard factual quality gating.
- Updated Task 3 (`mogemma-yj1.2.3`) to enforce multi-model generation validation outcomes.
- Added Task 4 (`mogemma-yj1.2.4`) for nano tensor mapping/layout invariants.

### Impact Assessment

- Tasks affected: mogemma-yj1.2.1, mogemma-yj1.2.2, mogemma-yj1.2.3, mogemma-yj1.2.4
- Timeline impact: Moderate, due to deeper nano path analysis
- Dependencies updated: Added new downstream chapter epic for embedding model-id policy (`mogemma-yj1.4`)

## [2026-02-24 20:30] Revision 4

**Type:** both
**Reason:** Root-cause research confirmed nano gibberish is caused by architecture-semantic drift (not broken weights). Plan needed to pivot from shape-only hardening to reference-faithful Gemma3n implementation, and to add runtime performance work for Mojo parameter storage/config.

### Changes Made

**Spec Changes:**
- Reframed nano failure as semantic parity gap versus Gemma3n reference behavior.
- Added explicit requirement for parity across AltUp, per-layer input, and attention/cache semantics.
- Added requirement to eliminate per-token model descriptor reconstruction by caching runtime descriptors at init.

**Plan Changes:**
- Updated Task 1 (`mogemma-yj1.2.1`) scope to full semantic parity implementation.
- Added `mogemma-yj1.2.5` for intermediate parity checks during nano refactor.
- Added `mogemma-yj1.2.6` and `mogemma-yj1.2.7` for Mojo runtime descriptor caching and hot-path rebuild removal.
- Added `mogemma-yj1.2.8` for lightweight performance regression checks.
- Kept `mogemma-yj1.2.4` as required conversion/layout guardrail.

### Impact Assessment

- Tasks affected: mogemma-yj1.2.1, mogemma-yj1.2.3, mogemma-yj1.2.4, mogemma-yj1.2.5, mogemma-yj1.2.6, mogemma-yj1.2.7, mogemma-yj1.2.8
- Timeline impact: Moderate to high; parity implementation is broader than isolated bug fixes but should produce stable correctness.
- Dependencies updated: No chapter-level dependency changes; new tasks added under `mogemma-yj1.2`.

## [2026-03-06 00:28] Revision 5

**Type:** both
**Reason:** Tightened the remaining active plan so a single-pass implementation PR can land without ambiguity, and made hot-path performance evidence an explicit acceptance requirement instead of an implied goal.

### Changes Made

**Spec Changes:**
- Added a primary file/evidence target section so the PR scope is anchored to the actual runtime, test, and validation surfaces.
- Added explicit acceptance requirements for steady-state descriptor-build counts, public API preservation, fail-fast behavior, and required correctness/performance evidence.

**Plan Changes:**
- Added a frozen checkpoint-set task for intermediate nano parity.
- Added explicit runtime-state caching scope for prepared layer views, RoPE tables, and variant constants.
- Added a manual warm-vs-cold verification task covering semantic behavior and descriptor-build stability.

### Impact Assessment

- Tasks affected: mogemma-yj1.2.1, mogemma-yj1.2.3, mogemma-yj1.2.5, mogemma-yj1.2.6, mogemma-yj1.2.7, mogemma-yj1.2.8
- Timeline impact: Moderate; the changes reduce implementation ambiguity and should lower rework risk.
- Dependencies updated: None
