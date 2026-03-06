# Master PRD: Fix End-to-End Inference (Generation & Embeddings)

## Context
`core-embedding-padding` is already complete, but production validation still shows three unresolved concerns:
1. Generation quality is unstable (`gemma3n-e2b-it` emits gibberish).
2. The standard model currently lacks semantic quality gates (for example, "capital of France" quality checks).
3. Validation includes an embedding model id that is not resolvable from the public `gemma-data` bucket.

This PRD extension focuses on restoring reliable generation behavior and enforcing deterministic, explicit validation contracts without soft fallbacks.

## Roadmap
1. **Chapter 1: `core-generation-fix`**
   - Root-cause and resolve gibberish generation, prioritizing the nano path (`gemma3n-e2b-it`) in Mojo core math and tensor mapping.
   - Add deterministic semantic generation checks for standard Gemma model behavior.
2. **Chapter 2: `validation-enhancements`**
   - Expand `tools/validate.py` with semantic assertions (3-5 prompts), deterministic decoding gates, and strict failure diagnostics.
3. **Chapter 3: `embedding-model-id-policy`**
   - Remove non-existent default embedding model IDs from baseline validation coverage.
   - Keep explicit fail-fast errors for requested but unresolvable model IDs.

### Completed Foundation
- `core-embedding-padding` (beads: `mogemma-yj1.1`) is complete and remains a prerequisite for this active roadmap.

## Global Constraints
- **Performance First**: Fixes in the core engine (Mojo) are preferred over Python-level workarounds.
- **Maintainability**: Ensure consistent types and clear logic when traversing the Mojo <-> Python bridge.
- **Deterministic Contracts**: Generation quality gates must run under controlled decoding settings (including deterministic mode where needed).
- **Error Transparency**: No soft fallbacks. Runtime and validation must fail explicitly with actionable errors.
- **Validation**: Any code change must pass `tools/validate.py`.
