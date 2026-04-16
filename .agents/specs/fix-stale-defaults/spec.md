# Flow: fix-stale-defaults

## Specification

### Goal

Make the public mogemma defaults work out-of-the-box against `gs://gemma-data`
and remove every stale `gemma3-*` / `gemma3n-*` reference left behind by the
hub migration. No new functionality — only unblocking + honesty about what is
currently executable.

### Parent

No parent PRD. Lands directly on `fix/timeout` (PR #16) per user direction on
2026-04-16 — the separate-branch option was considered and rejected because
`origin/main` is 37 commits behind `fix/timeout`, lacks the entire `.agents/`
Flow infrastructure, and carries its *own* broken `EmbeddingConfig` default
(`"google/gemma-4-26B-A4B-it"` — 52 GB of MoE as the embedding default).
Stacking a second PR off `main` would have stranded this work without
context and forked the bug story. These fixes therefore ship as commits on
PR #16.

### Problem statement

A live probe of `gs://gemma-data` on 2026-04-16 shows the following gap
between code defaults and actual bucket contents:

| Code reference | Resolves to | Bucket state |
|---|---|---|
| `EmbeddingConfig.model_path = "google/gemma-4-E4B"` | `checkpoints/gemma4-e4b/` | **0 objects** |
| `GenerationConfig.model_path = "google/gemma-4-E4B-it"` | `checkpoints/gemma4-e4b-it/` | 29 objects, 28.91 GB ✅ |
| `tools/smoke_test.py` default `"gemma3n-e2b-it"` | `checkpoints/gemma3n-e2b-it/` | **prefix does not exist** |
| `tools/run_parity_gates.py` default `"gemma3n-e2b-it"` | same | **prefix does not exist** |
| `tools/validate.py` `TEXT_MODEL_ID = "gemma3-270m-it"` | `checkpoints/gemma3-270m-it/` | **prefix does not exist** |
| `tools/validate.py` `NANO_MODEL_ID = "gemma3n-e2b-it"` | same | **prefix does not exist** |
| `.github/workflows/ci.yml:177` `--model gemma3n-e2b-it` | same | **prefix does not exist** |

Net effect:
- Instantiating `EmbeddingModel()` with defaults raises
  `No objects found under gs://gemma-data/checkpoints/gemma4-e4b/`.
- `tools/smoke_test.py`, `tools/run_parity_gates.py`, `tools/validate.py` all
  crash before the first inference call.
- The `check-release` CI job (workflow_dispatch only) would crash on the
  parity-gate step if triggered.

### Non-goals

1. End-to-end inference validation. Real-weight tests are gated on the
   follow-on `gpu-ci-modal` flow (Tier C) and/or an ephemeral GCP VM flow.
   Local runs require ~36 GB for even the smallest working model
   (`gemma-4-E2B-it`) which the current workstation cannot hold.
2. `src/mo/tests` → CI. Separate follow-on.
3. Metal `DeviceContext()` arch-string workaround. Separate follow-on.
4. Adding pretrained E2B / E4B to the GCS bucket. Out of scope for this
   repo; track as an upstream ask.

### Models currently available in `gs://gemma-data` (as of 2026-04-16 probe)

- `checkpoints/gemma4-e2b-it/` — 24 objects, 18.25 GB (Orbax)
- `checkpoints/gemma4-e4b-it/` — 29 objects, 28.91 GB (Orbax)
- `checkpoints/gemma4-26b-a4b-it/` — 44 objects, ~52 GB (Orbax)

Only the instruction-tuned (`-it`) variants and the MoE 26B exist in the
bucket. Pretrained E2B / E4B are **not** present.

### Design decision — resolved 2026-04-16

**`EmbeddingConfig` default: Option A — `"google/gemma-4-E4B-it"`.**

Rationale captured at decision time:
- Pretrained E4B is not in `gs://gemma-data` (0 objects). Can't ship a
  default that crashes.
- Quality delta vs. pretrained is small (typically 1–3% on MTEB-style
  retrieval benchmarks). Acceptable in exchange for "it works."
- Docstring must note pretrained is the ideal but unavailable, and that
  the default will flip back when pretrained variants are published to
  the bucket.

Options B (remove default) and C (improve error message, keep pretrained)
were considered and rejected as, respectively, a gratuitous API break and
friendly-crash theater.

---

## Implementation Plan

### Phase 0: Preparation

- [x] **0.1 GCS probe reproduced 2026-04-16.** `checkpoints/gemma4-e4b/` → 0 objects; `checkpoints/gemma4-e2b/` → 0 objects; `checkpoints/gemma3n-e2b-it/` → prefix does not exist. Problem statement is accurate.

### Phase 1: Unit-testable fixes (no network, no storage)

- [ ] **1.1 Fix `EmbeddingConfig` default per user decision.**
  - File: `src/py/mogemma/config.py:19`, docstring lines 20–25.
  - Action depends on user choice of Option A / B / C (see "Design decision pending").
  - **Test-first:** `src/py/tests/test_config.py` — add `test_embedding_config_default_is_concrete_and_in_catalog` that resolves `EmbeddingConfig().model_path` through `HubManager._clean_model_id` + `_gcs_checkpoint_prefix` and asserts the resulting prefix is in a maintained allow-list of known-good prefixes. (Allow-list lives as a module-level constant in `hub.py` — see 1.6.)

- [ ] **1.2 Fix `tools/smoke_test.py`.**
  - File: `tools/smoke_test.py:17`.
  - Default → `"google/gemma-4-E2B-it"` (smallest working model; 18.25 GB is still big but this is just the default; env var override preserved).
  - **Test-first:** `src/py/tests/test_tools_defaults.py::test_smoke_test_default_in_catalog`.

- [ ] **1.3 Fix `tools/run_parity_gates.py`.**
  - File: `tools/run_parity_gates.py:67`.
  - Default → `"google/gemma-4-E2B-it"`.
  - **Test-first:** `src/py/tests/test_tools_defaults.py::test_run_parity_gates_default_in_catalog`.

- [ ] **1.4 Rewrite `tools/validate.py` constants and dead branches.**
  - File: `tools/validate.py` lines 17–19, 32–35, 108–109.
  - `TEXT_MODEL_ID` → `"google/gemma-4-E4B-it"`.
  - `EMBED_MODEL_ID` → same as the final `EmbeddingConfig` decision (task 1.1).
  - `NANO_MODEL_ID` → `"google/gemma-4-E2B-it"`.
  - Delete the `if "gemma3n" in model_id: return` branch in `_assert_semantic_quality` (lines 34–35) — we no longer ship any gemma3n model.
  - **Test-first:** `src/py/tests/test_tools_defaults.py::test_validate_constants_in_catalog` (asserts each of the three IDs is in the catalog).

- [ ] **1.5 Fix `.github/workflows/ci.yml` `check-release` parity step.**
  - File: `.github/workflows/ci.yml:177`.
  - Decision: **remove the parity-gate step from `check-release`** and replace it with a clear one-line comment noting that real-weight parity is gated on the follow-on `gpu-ci-modal` flow. Rationale: (a) standard Linux runners have ~14 GB disk, smallest model needs ~36 GB; (b) CPU-only inference on E2B-it would blow past the 15-minute job timeout; (c) keeping a known-broken step reduces release-runbook trust. The artifact-upload for `parity_artifact.json` is removed in the same edit.
  - **Test-first:** `src/py/tests/test_ci_workflow.py::test_no_gemma3_references_in_ci_yml` — parse the YAML file, assert that no string starting with `gemma3-` or `gemma3n-` exists anywhere in it.

- [ ] **1.6 Introduce a single catalog of known-good model IDs.**
  - File: `src/py/mogemma/hub.py` (new module-level constant after `_GCS_TOKENIZER_PATH`).
  - Shape:
    ```python
    KNOWN_GCS_MODELS: frozenset[str] = frozenset({
        "google/gemma-4-E2B-it",
        "google/gemma-4-E4B-it",
        "google/gemma-4-26B-A4B-it",
    })
    ```
  - Exported from `mogemma.hub` for test imports. Used by the test suite as
    a static allow-list; not enforced at runtime (a typo in `model_path`
    should still surface as a clean GCS 404, not a KeyError).
  - **Test-first:** `src/py/tests/test_hub.py::test_known_gcs_models_frozenset_nonempty`.

### Phase 2: Repo-wide grep gate (regression prevention)

- [ ] **2.1 Kill every live `gemma3-` / `gemma3n-` reference outside of historical specs/learnings.**
  - Whitelist (expected residual matches):
    - `.agents/specs/*/spec.md` (historical)
    - `.agents/knowledge/learnings.md` (historical)
    - `.agents/specs/archive/**` (archived)
  - Any match outside that whitelist fails the test.
  - **Test-first:** `src/py/tests/test_no_stale_model_refs.py::test_no_live_gemma3_references` — walks the repo under `src/`, `tools/`, `.github/`, asserts no match.

### Phase 3: Live-network verification (optional, opt-in)

- [ ] **3.1 Add a network-gated smoke that actually lists each catalog entry.**
  - File: `src/py/tests/test_hub_live.py` (new).
  - Marked `@pytest.mark.gcs` and skipped unless `MOGEMMA_LIVE_GCS=1` is set.
  - For each entry in `KNOWN_GCS_MODELS`, `obs.list(store, prefix=...)` asserts ≥1 object.
  - Not wired to CI by default; a follow-up flow can wire this to a weekly cron if desired.

### Phase 4: Documentation

- [ ] **4.1 Update `README.md`** — any "quick start" example must use an
  available model ID. No code example should reference E4B pretrained.
- [ ] **4.2 Update `docs/release-runbook.md`** — explain that the
  parity-gate step has been removed from `check-release` and will return
  via the `gpu-ci-modal` Tier C flow.

### Verification Gate

- [ ] `make lint` clean.
- [ ] `make test` green on Linux CPU (no real downloads).
- [ ] `rg -n 'gemma3[n-]|gemma3-' src/ tools/ .github/` returns 0 live hits.
- [ ] `uv run python -c "from mogemma import EmbeddingConfig, GenerationConfig; EmbeddingConfig(); GenerationConfig()"` exits 0 (defaults construct without error; no network call needed for construction itself).
- [ ] Optional: `MOGEMMA_LIVE_GCS=1 uv run pytest src/py/tests/test_hub_live.py -v` passes on a network-connected machine.

### Risks & known unknowns

1. **The `EmbeddingConfig` default decision has product-level implications.**
   Pretrained is empirically better for embeddings. Switching to `-it`
   (Option A) is a silent quality regression for any caller relying on the
   default; switching to "no default" (Option B) is a breaking API change.
   User must choose.
2. **Removing the `check-release` parity step feels like a retreat.** It is —
   but keeping a step that crashes if triggered is worse. Replaced by a
   follow-on flow, not abandoned.
3. **The catalog constant `KNOWN_GCS_MODELS` risks becoming stale.** Phase
   3.1 live-network smoke is the mitigation; wiring it to a weekly cron in
   a follow-up would keep it honest.

### Follow-on flows

- `gpu-ci-modal` — Tier C. Modal GPU runner for `test_gpu_*.mojo` +
  on-demand real-weight parity. User already expressed preference for Modal.
- `mojo-tests-in-ci` — Tier 0 item #1 from earlier plan. Add `src/mo/tests`
  to CI pytest target for CPU-path Mojo regression coverage.
- `metal-devicecontext-workaround` — investigate the
  `metal:1-metal4` stdlib arch-list rejection.
- `vision-gpu-null-ptr-fix` — fix the vision-encoder null-ptr rebind in
  the GPU upload path that was gated out in `657f8ed`.
