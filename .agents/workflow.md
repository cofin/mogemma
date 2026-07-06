# Project Workflow

<!-- truth: start -->

## Guiding Principles

1. **Beads backend is the Source of Truth:** Prefer official Beads (`bd`), keep `br` as compatibility mode, and use `/flow:sync` to export task state to `spec.md` when needed.
2. **The Tech Stack is Deliberate:** Changes to the tech stack must be documented in `tech-stack.md` *before* implementation.
3. **Test-Driven Development:** Write unit tests before implementing functionality.
4. **High Code Coverage:** Aim for >80% code coverage for all modules.
5. **Developer Experience First:** mogemma is a library — every decision should prioritize the Python developer consuming the public API.
6. **Non-Interactive & CI-Aware:** Prefer non-interactive commands. Use `CI=true` for watch-mode tools (tests, linters) to ensure single execution.
7. **Use the Repo's Real Commands:** Prefer canonical project entrypoints — `make lint`, `make test`, `make check-all`, `make build`, `make coverage`, `make benchmark` — before inventing ad hoc commands.
8. **Be Collaborative:** Never use blamey or ownership-deflecting language such as "not my issue" or "not caused by my change." Describe unrelated failures factually, offer the smallest useful next step, and ask the user whether to handle them now or separately.
9. **Minimal Targeted Changes:** Make the smallest coherent change set that solves the task. Do not make opportunistic cleanup edits or unrelated modifications without approval.
10. **No Silent Descoping:** If the task is larger or messier than expected, refine the plan or ask the user how to prioritize. Never stub with `NotImplementedError` to make a task appear complete.

## Beads Integration

Flow supports three modes:

- Official Beads (`bd`) — preferred default
- beads_rust compatibility (`br`)
- No Beads — degraded mode for docs/plans/lightweight local work

Configured for local-only use unless the user explicitly asks for shared repo state.

### Session Protocol

**Session Start:** Use the active backend's session-start commands (`bd ready` / `br ready`) to pick up unblocked work.

**Session End:** For local-only ignores, prefer `.git/info/exclude` before `.gitignore`. Sync notes with the active backend and commit `.beads/` state when using `br`.

> If no supported Beads backend is available, workflow degrades gracefully to git-only tracking.

### When to Track in Beads

**Rule: If work takes >5 minutes, track it in Beads.**

| Duration | Action | Example |
|----------|--------|---------|
| <5 min | Just do it | Fix typo, update config |
| 5-30 min | Create task | Add validation, write test |
| 30+ min | Create task with subtasks | Implement feature |

**Why this matters:**

- Notes survive context compaction — critical for multi-session work.
- The active Beads backend finds unblocked work automatically.
- If resuming in 2 weeks would be hard without context, use Beads.

### Creating Issues with Full Context

**CRITICAL:** Always include the backend's purpose/description field at creation time, then add context notes separately.

```bash
# See `choosing-beads-backend` for exact command mapping.
# bd/br: create with --description, then update with --notes/comment.
```

- `--description`: Purpose and goal.
- notes/comments: Context for future agents.
- Priority levels: `P0`=critical, `P1`=high, `P2`=medium, `P3`=low, `P4`=backlog.

## Task Workflow

All tasks follow a strict lifecycle.

### Task Workflow (TDD) — Beads-First

**CRITICAL:** Beads is the source of truth. Never write `[x]`, `[~]`, `[!]`, or `[-]` markers to `spec.md` manually. After ANY Beads state change, agents MUST run `/flow:sync` to update `spec.md`.

**Companion Skills Usage:**

- **Analysis:** Use `flow:tracer` for systematic code exploration before implementation.
- **Design:** Use `flow:consensus` when choosing between multiple implementation approaches.
- **Validation:** Use `flow:challenge` when reviewing claims to prevent reflexive agreement.
- **Debugging:** Use `flow:deepthink` if a problem resists quick answers or investigation goes in circles.
- **External Docs:** Use `flow:apilookup` for authoritative API/framework docs, versions, breaking changes.
- **Security:** Use `flow:security-auditor` when touching auth, input handling, secrets, or API keys.
- **Architecture:** Use `flow:architecture-critic` when adding modules, changing boundaries, or assessing coupling.
- **Performance:** Use `flow:performance-analyst` for hot paths, GPU kernels, caching, FFI overhead.
- **Multiple Views:** Use `flow:perspectives` when weighing trade-offs or evaluating decisions.
- **Pushback:** Use `flow:devils-advocate` during PR review or when a decision lacks visible opposition.
- **Documentation:** Use `flow:docgen` when generating API docs, module docs, or reference guides.
- **Domain Skills:** Consult `patterns.md` Skill Associations table for language, framework, and domain-specific skills.

1. **Select Task:** Use the active backend's ready queue, or fall back to parsing `spec.md`.

2. **Mark In Progress:**
   - Sync to Beads using the active backend.
   - **Do NOT edit `spec.md`** — Beads is source of truth.

3. **Write Failing Tests (Red Phase):**
   - Create a new test file under the appropriate boundary:
     - Python: `src/py/tests/unit/` or `src/py/tests/integration/`
     - Mojo: `src/mo/tests/unit/` or `src/mo/tests/integration/`
   - Write one or more unit tests that clearly define the expected behavior and acceptance criteria.
   - **CRITICAL:** Run the tests and confirm they fail as expected. Do not proceed until you have failing tests.

4. **Implement to Pass Tests (Green Phase):**
   - Write the minimum amount of code necessary to make the failing tests pass.
   - For Mojo edits, run `make build` to recompile `mogemma._core` before re-running pytest.
   - Rerun the test suite and confirm all tests pass.

5. **Refactor (Optional but Recommended):**
   - With passing tests as a safety net, improve clarity and remove duplication without changing behavior.
   - Rerun tests after refactoring.

6. **Verify Coverage:** Run the project's coverage target:

   ```bash
   make coverage
   ```

   Target: >80% coverage for new code. Output lands in `htmlcov/index.html`.

7. **Document Deviations:** If implementation differs from the documented tech stack:
   - **STOP** implementation.
   - Update `tech-stack.md` with the new design.
   - Add a dated note explaining the change.
   - Resume implementation.

8. **Commit Code Changes:**
   - Stage changes related to the task.
   - Propose a conventional-commits message, e.g. `feat(convert): add sharded safetensors writer`.
   - Perform the commit.

9. **Record Task Completion (Beads-First):**
   - **Step 9.1: Get Commit Hash:** `git log -1 --format="%h"`.
   - **Step 9.2: Close in Beads:** Use the active backend's close command with `--reason "commit: <sha>"`.
   - **Step 9.3 (Manual Sync):** Run `/flow:sync` to update `spec.md` markers from Beads state.
   - **Do NOT manually edit `spec.md` markers.**

10. **Log Learnings:**
    - Append discoveries to the flow's `learnings.md` (under `.agents/specs/<flow-id>/learnings.md`).
    - Sync to Beads using the active backend's note/comment command.
    - Elevate reusable patterns to `.agents/patterns.md` at phase completion.
    - Elevate component-level facts to the matching file in `.agents/knowledge/`.
    - If the user had to repeat a correction or showed frustration, capture that as a workflow gap and elevate it into the knowledge system.
    - Capture validated repo-native commands so future agents reuse the same `make` entrypoints.
    - If `.agents/skills/flow-memory-keeper/SKILL.md` exists, update it with durable project-specific refinements.

### Knowledge Flywheel

1. **Capture** — After each task, append learnings to the flow's `learnings.md`.
2. **Elevate** — At phase/flow completion, move reusable patterns to `.agents/patterns.md`.
3. **Synthesize** — Integrate learnings directly into the existing component chapters under `.agents/knowledge/` (`architecture.md`, `gemma4-models.md`, `python-runtime.md`, `mojo-runtime.md`, `gpu-infrastructure.md`, `build-and-packaging.md`, `ci-and-release.md`, `quality-gates.md`, `performance.md`, `learnings.md`). Update current state; do NOT add per-flow files.
4. **Inherit** — New flows read `patterns.md` + scan `.agents/knowledge/` chapters.

Repeated user corrections and validated repo-native commands are both high-signal learning triggers. Do not leave them buried in chat history.

**Knowledge Base:**

| Tier | File | Loaded | Purpose |
|------|------|--------|---------|
| **Patterns** | `.agents/patterns.md` | Always | Elevated actionable rules for priming |
| **Knowledge Chapters** | `.agents/knowledge/*.md` | On demand | Synthesized implementation details and current state |
| **Style Guides** | `.agents/code-styleguides/*.md` | On demand | Language-specific house style |

**Important:** `.agents/patterns.md` is NOT archived with flows. It remains at the top level as persistent project knowledge. Knowledge chapters in `.agents/knowledge/` also persist independently of archives and describe the active codebase state.

**Learnings Entry Format:**

```markdown
## [YYYY-MM-DD HH:MM] - Phase N Task M: Task Description

- **Implemented:** Brief description
- **Files changed:** path/to/files
- **Commit:** abc1234
- **Learnings:**
  - Patterns: Codebase uses X for Y
  - Gotchas: Must do Z before W
  - Context: Module A owns B
```

### Phase Completion Verification and Checkpointing Protocol

**Trigger:** This protocol runs immediately after a task that also concludes a phase in `spec.md`.

1. **Announce Protocol Start:** Inform the user that the phase is complete and the verification and checkpointing protocol has begun.

2. **Ensure Test Coverage for Phase Changes:**
   - **Step 2.1: Determine Phase Scope:** Read `spec.md` to find the Git SHA of the previous phase's checkpoint. If none, scope is all changes since the first commit.
   - **Step 2.2: List Changed Files:** `git diff --name-only <previous_checkpoint_sha> HEAD`.
   - **Step 2.3: Verify and Create Tests:** For each code file (exclude `.json`, `.md`, `.yaml`):
     - Verify a corresponding test file exists under the appropriate Python or Mojo `unit/` / `integration/` tree.
     - If missing, analyze existing test files for naming + style, then write tests validating the phase's `spec.md` tasks.

3. **Execute Automated Tests with Proactive Debugging:**
   - Announce the exact command first.
   - **Example Announcement:** "I will now run the automated test suite to verify the phase. **Command:** `CI=true make check-all`"
   - Execute.
   - On failure: inform the user, propose a fix **at most twice**. If still failing, stop and ask for guidance.

4. **Propose a Detailed, Actionable Manual Verification Plan:**
   - Analyze `product.md`, `product-guidelines.md`, and `spec.md` for user-facing goals of the completed phase.
   - Generate a step-by-step plan with commands and expected outcomes.

     **For a Python API Change:**

     ```
     The automated tests have passed. For manual verification, please follow these steps:

     **Manual Verification Steps:**
     1.  **Start a Python REPL with the dev env:** `uv run python`
     2.  **Import and exercise the new API:**
         ```python
         from mogemma import SyncGemmaModel
         model = SyncGemmaModel("google/gemma-4-e2b-it")
         print(model.generate("hello"))
         ```
     3.  **Confirm that you see:** Non-empty generated text with no tracebacks, and the model prints from the local cache if already downloaded.
     ```

     **For a Mojo/Kernel Change:**

     ```
     The automated tests have passed. For manual verification, please follow these steps:

     **Manual Verification Steps:**
     1.  **Rebuild the extension:** `make build`
     2.  **Run the smoke test:** `make smoke-test`
     3.  **Confirm that you see:** `smoke-test: ok` with no segfaults or FFI errors.
     ```

     **For a Conversion/Loader Change:**

     ```
     The automated tests have passed. For manual verification, please follow these steps:

     **Manual Verification Steps:**
     1.  **Clear the relevant cache slice:** `rm -rf ~/.cache/mogemma/google/<variant>`
     2.  **Trigger a cold download + conversion:** `uv run python -c "from mogemma.hub import HubManager; HubManager().download_sync('google/gemma-4-e2b-it')"`
     3.  **Confirm that you see:** A `safetensors` directory + `config.json` in the cache path, and no Orbax residue (OCDBT files cleaned up per hub.py policy).
     ```

5. **Await Explicit User Feedback:**
   - After presenting the plan, ask: "**Does this meet your expectations? Please confirm with yes or provide feedback on what needs to be changed.**"
   - **PAUSE** and await an explicit confirmation.

6. **Create Checkpoint Commit:**
   - Stage all changes; create an empty commit if none.
   - Message example: `flow(checkpoint): Checkpoint end of Phase X`.

7. **Record Verification in Beads:**
   - Update the epic with a verification summary using the active backend's note/comment command.

8. **Sync to `spec.md` (Manual):**
   - Run `/flow:sync` to align `spec.md` markers with Beads state.
   - **Do NOT manually edit `spec.md`.**

9. **Announce Completion:** Inform the user that the phase is complete and the checkpoint has been recorded in Beads.

### Quality Gates

Before marking any task complete, verify:

- [ ] All tests pass (`make test`)
- [ ] Code coverage meets requirements (`make coverage` ≥ 80%)
- [ ] Code follows project style guides (`.agents/code-styleguides/python.md`, `mojo.md`)
- [ ] All public functions/methods have Google-style docstrings
- [ ] Type safety enforced (mypy + pyright both pass on Python; Mojo type-checks clean)
- [ ] No linting or static analysis errors (`make lint`)
- [ ] Canonical repo verification commands from this workflow were used (no ad hoc invocations)
- [ ] Documentation updated if public API or component behavior changed (matching `.agents/knowledge/` file)
- [ ] No security vulnerabilities introduced (see Code Review §Security)
- [ ] No new `# noqa` / `# type: ignore` without rule code + reason comment

## Development Commands

**Canonical entrypoints for mogemma.** Prefer these over raw `ruff` / `pytest` / `hatch` invocations; CI runs the same targets.

### Setup

```bash
make install                 # uv venv + uv sync --all-extras --dev (first-time setup)
make build                   # hatch-mojo compile core.mojo → mogemma._core.so
```

`make build` is required after any Mojo source edit before `make test` can see the change.

### Daily Development

```bash
make lint                    # ruff check + ruff format --check + mojo format
make test                    # pytest (src/py/tests + src/mo/tests)
make type-check              # mypy + pyright
make coverage                # pytest with coverage → htmlcov/index.html
make smoke-test              # Mojo bridge smoke test
make benchmark               # deterministic release benchmark harness (advisory)
```

### Before Committing

```bash
make check-all               # lint + test + coverage — canonical pre-PR gate
```

For release preflight: `make check-release` (adds benchmarks + release-only checks).

## Testing Requirements

### Unit Testing

- Every module must have corresponding tests. Python tests live under `src/py/tests/unit/` or
  `src/py/tests/integration/`; Mojo tests live under `src/mo/tests/unit/` or `src/mo/tests/integration/`.
- Use pytest fixtures for setup/teardown. Mock external dependencies where appropriate.
- Test both success and failure cases.
- Deterministic fixtures only — no unseeded random init for parity testing.

### Integration Testing

- Test complete user flows: download → load → generate / embed.
- Verify conversion round-trips (Orbax → safetensors → load → forward-pass).
- Test async streaming via `anyio`.

### Contract Testing

- Cross-language (Python ↔ Mojo) boundaries tested via `test_altup_contract.mojo`-style contract tests.
- FFI shape/dtype contracts validated at both sides.

### Mobile Testing

N/A — mogemma is a Python/Mojo library with no mobile surface.

## Code Review Process

### Self-Review Checklist

Before requesting review:

1. **Functionality**
   - Feature works as specified
   - Edge cases handled (short/long inputs, empty batches, large batches)
   - Error messages are developer-friendly and distinguish Python- vs Mojo-layer failures

2. **Code Quality**
   - Follows `.agents/code-styleguides/python.md` or `mojo.md`
   - DRY principle applied
   - Clear variable/function names
   - Comments only for non-obvious WHY, not WHAT

3. **Testing**
   - Unit tests comprehensive
   - Contract tests cover FFI boundaries
   - Coverage adequate (>80% on touched modules)
   - No placeholder stubs that pass via default return

4. **Security**
   - No hardcoded secrets / API keys
   - Input validation at FFI boundary (shape, dtype, contiguity)
   - No command/shell injection in any subprocess calls
   - No path traversal in cache / local I/O
   - SQL injection / XSS: N/A (no DB, no web surface)

5. **Performance**
   - Hot paths avoid per-call allocations
   - GPU code gated by `@comptime if has_accelerator()`
   - FFI pointers reused, not re-extracted per loop iteration
   - Streaming patterns preferred over eager loading for multi-GB checkpoints

6. **Mobile Experience**

   N/A for mogemma (library, no mobile surface).

## Commit Guidelines

### Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Enforced by `.github/workflows/pr-title.yml`. Keep titles ≤70 characters; put detail in the body.

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Performance improvement
- `test`: Adding missing tests
- `build`: Build system / packaging changes
- `ci`: CI configuration changes
- `chore`: Maintenance tasks

### Examples

Recent mogemma history:

```bash
git commit -m "feat(inventory): add script to dump Orbax tensor inventory for Gemma 4 variant"
git commit -m "chore(checkpoint): Phase 1 complete (Orbax→safetensors scaffold)"
git commit -m "feat(convert): add sharded safetensors writer"
git commit -m "chore(convert): satisfy lint + type-check gates"
git commit -m "feat(convert): implement base-transformer Orbax→safetensors iterator"
```

## Definition of Done

A task is complete when:

1. All code implemented to specification.
2. Unit tests written and passing.
3. Code coverage meets project requirements (>80% on touched modules).
4. Documentation complete: matching `.agents/knowledge/` file updated if public API or component behavior changed.
5. Code passes all configured linting and static analysis checks (`make lint`).
6. Works correctly on supported platforms (Linux x86_64/aarch64, macOS x86_64/arm64, Python 3.10–3.13).
7. Implementation notes added to `spec.md` for the relevant task.
8. Changes committed with a conventional-commits message.
9. Task closed in Beads with the active backend's completion command and commit reference.
10. Markdown synced manually by running `/flow:sync`.
11. No ignored Flow artifacts were force-added to git.

## Emergency Procedures

### Critical Bug in a Published Release

1. Create hotfix branch from `main`.
2. Write failing test for the bug.
3. Implement minimal fix.
4. Run `make check-all` locally.
5. Open PR; once merged, `make release bump=patch` to publish a patch version via `publish.yml`.
6. Document in `spec.md` and note in `.agents/knowledge/learnings.md` if relevant.

### Data Loss

mogemma is a library — it does not manage user databases. The analog here is **cache corruption** (`~/.cache/mogemma/<model-id>`):

1. Stop any in-flight downloads (Ctrl-C to abort `HubManager.download_sync`).
2. Rename the corrupted cache slice rather than deleting (for forensics): `mv ~/.cache/mogemma/<id> ~/.cache/mogemma/<id>.corrupt`.
3. Re-download cleanly.
4. If corruption pattern is reproducible, file a Beads issue with the cache listing + any `convert.py` / `hub.py` logs.

### Security Breach

For a credential/secret leak in the repo:

1. Rotate all affected secrets immediately (PyPI trusted publishing tokens, any GCS/HF auth headers in tests).
2. Review access logs where applicable.
3. Patch the leak; force-push only if the user explicitly approves (history rewrite).
4. Document the incident in `.agents/knowledge/learnings.md` under "Hard-won learnings".

## Deployment Workflow

mogemma "deployment" is a PyPI release. See `.agents/knowledge/ci-and-release.md` for the workflow structure.

### Pre-Release Checklist

- [ ] `make check-all` passes locally
- [ ] `make check-release` passes (includes benchmarks)
- [ ] `CHANGELOG.md` or release notes drafted
- [ ] Version bumped via `bump-my-version` — never hand-edit version strings
- [ ] No new `# noqa` / `# type: ignore` in touched files without justification
- [ ] `.agents/knowledge/` chapters updated for any public-API or component changes

### Release Steps

```bash
make release bump=patch      # bump, commit, tag, push
# Pre-release:
make pre-release version=0.2.0-alpha.1
```

1. Tag push triggers `publish.yml`.
2. `publish.yml` builds wheels (Linux x86_64/aarch64, macOS x86_64/arm64) and publishes to PyPI via OIDC trusted publishing.
3. Verify the release on PyPI and install into a clean venv to sanity-check.

### Post-Release

1. Monitor PyPI download stats and GitHub issue tracker.
2. Check CI dashboards for regressions on downstream dependency updates.
3. Gather feedback from developer consumers.
4. Plan next iteration; Beads backlog (`bd ready` / `br ready`) drives prioritization.

## Continuous Improvement

- Review workflow regularly; update based on pain points.
- Document lessons learned in the flow's `learnings.md`, then elevate to `.agents/patterns.md` or `.agents/knowledge/<component>.md`.
- Capture user corrections, missing defaults, and canonical repo commands so they stop being chat-only reminders.
- Optimize for developer happiness (the downstream Python developer consuming `mogemma`).
- Keep things simple and maintainable — complexity belongs in the runtime, not the API.
<!-- truth: end -->
