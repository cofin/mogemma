# Flow: gpu-nano-parity-gates_20260304

## Flow Context
- Flow ID: `gpu-nano-parity-gates_20260304`
- Status: `planned`
- Parent PRD: `mojo-gpu-nano-parity_20260304`
- Chapter: `3` (Deterministic parity + performance acceptance gates)
- Beads Epic: `mogemma-13d.3`

## Local Status Snapshot

1. Beads is currently unreliable in this workspace, so local spec artifacts are the current source for implementation status.
2. This flow remains unimplemented locally.
3. Upstream GPU device-selection Phase 2 is now implemented locally in `gpu-device-selection-api_20260304`, so gate design can assume the normalized device descriptor is the current Python-side contract.
4. No nano parity-gate harness, fixtures, or threshold wiring has been applied for this flow yet.

## Specification
Define strict, deterministic nano correctness and quality gates for CPU vs GPU behavior, plus performance gates that prevent regressions while chapter 1/2 work lands.

### Code Analysis Summary
Files examined:
- `src/py/mogemma/model.py`
- `src/py/mogemma/config.py`
- `src/py/mogemma/convert.py`
- `src/mo/mogemma/core.mojo`
- `src/mo/mogemma/layers.mojo`

Key findings:
1. Deterministic generation gating must control sampling in Python (`_sample_next_token`) and not rely on Mojo-side sampling, because `step_mojo` returns logits only (`src/py/mogemma/model.py:103`, `src/mo/mogemma/core.mojo:366`).
2. Current generation loop already exposes deterministic controls via `GenerationConfig.temperature/top_k/top_p`; parity gates should standardize these settings (`src/py/mogemma/config.py:63`, `src/py/mogemma/model.py:355`).
3. Instruction-tuned prompt wrapping and EOS alias handling are part of semantic quality and must be included in parity fixtures (`src/py/mogemma/model.py:71`, `src/py/mogemma/model.py:169`).
4. Nano runtime carries explicit `arch`, `kv_share_start`, and `per_layer_dim` fields that are required metadata for parity diagnostics (`src/mo/mogemma/core.mojo:283`, `src/mo/mogemma/core.mojo:312`, `src/mo/mogemma/core.mojo:361`).
5. Conversion-time layout validation (`_validate_nano_layout`) prevents obvious shape corruption, but runtime parity gates are still required to detect semantic drift (`src/py/mogemma/convert.py:262`).

### Relevant Patterns
- Instruction-tuned Gemma prompts must use `<start_of_turn>/<end_of_turn>` wrapping to avoid low-quality artifacts (`.agent/patterns.md`).
- EOS handling for instruction models should prioritize `<end_of_turn>` to avoid marker drift (`.agent/patterns.md`).
- Deterministic validation (`temperature=0, top_k=1`) is mandatory for contract checks (`.agent/patterns.md`).
- Conversion validation alone is insufficient; runtime semantic checks are required (`.agent/patterns.md`).

### Required Gate Artifacts
- Structured parity artifact per run with: `prompt_id`, `seed`, `decode_config`, `arch`, `kv_share_start`, `per_layer_dim`, `backend`, `runner_class`, `mojo_version`, GPU/driver metadata, baseline id, and result summary.
- Fixture matrix covering short/medium/long prompts, EOS-sensitive prompts, KV-share-boundary prompts, and at least one known gibberish-regression prompt.
- Separate PR-smoke and release/default-enable reports so bounded CI coverage does not masquerade as release evidence.

### Requirements
#### Functional Requirements
1. Define deterministic nano parity gates for:
- token-level logits agreement (CPU vs GPU)
- generated token sequence agreement under deterministic decode settings
- quality sanity prompts (non-gibberish expectations)
2. Define required parity metadata capture per run (`arch`, `kv_share_start`, `per_layer_dim`, model id, seed, decode config).
3. Define benchmark gates for nano generation throughput relative to CPU baseline, with separate PR-smoke and release/default-enable thresholds.
4. Define failure taxonomy and reporting format so parity failures are actionable.
5. Freeze the deterministic decode profile in-spec: `temperature=0`, `top_k=1`, `top_p=1`, fixed prompt wrapping, fixed EOS policy, and fixed seed policy.
6. Freeze tolerance classes in-spec:
- logits/hidden parity: `atol <= 1e-4`, `rtol <= 1e-4`
- deterministic decode: exact token parity
- prompt-quality guardrails: no gibberish regression on the fixed sanity-prompt set
7. Default-enable performance eligibility starts at `>=1.25x` median throughput on medium/long prompts versus CPU baseline, while short-prompt p95 latency may not regress by more than `15%`.

#### Non-Functional Requirements
1. Gate runs must be reproducible and CI-friendly.
2. Gate runtime should be bounded and split into fast smoke + extended modes.
3. Gate thresholds must be explicit, versioned, and documented in flow artifacts.
4. Failure artifacts must be machine-readable enough for CI triage and release signoff.

#### Risks
1. Stochastic decode settings can mask regressions and produce flaky gates.
2. Prompt-format drift can create false parity failures unrelated to kernels.
3. Overly strict numeric tolerances can block valid GPU improvements; overly loose tolerances can hide true bugs.

## Implementation Plan
### Phase 1: Gate Contract Definition
- [ ] 1.1 Define the deterministic decode profile used by all nano parity tests (`temperature=0`, `top_k=1`, `top_p=1`, prompt wrapping, EOS policy, seed handling).
- [ ] 1.2 Define the required fixture matrix: short/medium/long prompts, EOS-sensitive prompts, KV-share-boundary prompts, and at least one known gibberish-regression prompt.
- [ ] 1.3 Define the exact threshold classes for logits/hidden parity, exact token parity, quality sanity checks, and release/default-enable throughput/latency policy.
- [ ] 1.4 Checkpoint: signed-off parity-gate spec with pass/fail rubric.

### Phase 2: TDD Gate Scaffolding
- [ ] 2.1 Add failing test harness for CPU-vs-GPU deterministic token comparison.
- [ ] 2.2 Add failing quality-gate tests for instruction-formatted prompts and EOS behavior.
- [ ] 2.3 Add failing performance gate scaffold with baseline capture and threshold checks.
- [ ] 2.4 Checkpoint: all new gate tests fail against unimplemented/incomplete GPU parity path.

### Phase 3: Gate Integration and Evidence Collection
- [ ] 3.1 Wire parity harness into validation workflow and emit the required artifact schema for every smoke/release run.
- [ ] 3.2 Capture initial CPU baselines, persist baseline identifiers, and produce triage-ready diff output for failing runs.
- [ ] 3.3 Document any accepted drift bounds, baseline waivers, and release-vs-smoke threshold policy in machine-readable and human-readable form.
- [ ] 3.4 Checkpoint: parity suite passes against reference implementation and emits structured artifacts with baseline ids and triage metadata.

### Phase 4: Manual Verification and Handoff Readiness
- [ ] 4.1 Run deterministic nano prompt set manually and review decoded outputs for coherence.
- [ ] 4.2 Run benchmark sample manually to validate throughput reporting pipeline.
- [ ] 4.3 Manual verification: confirm gate failure output includes actionable metadata (`kv_share_start`, decode settings, failing prompt id).
- [ ] 4.4 Checkpoint: chapter-ready parity package accepted for downstream GPU release gates.

## TDD-Oriented Task Checklist
- [ ] Red: create deterministic decode parity tests that fail if token stream diverges.
- [ ] Red: create instruction-prompt/EOS tests that fail on formatting regressions.
- [ ] Red: create benchmark gate tests that fail when throughput drops below agreed floor.
- [ ] Green: implement harness and thresholds until parity + perf gates pass.
- [ ] Refactor: normalize gate utilities/shared fixtures to reduce duplication.
- [ ] Regression: rerun full deterministic suite after every threshold or fixture change.
