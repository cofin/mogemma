# Tokenizer and Artifact Handoff Contract

## Objective

Define the exact request/response contract between the Python hub layer and the Rust+Mojo downloader boundary so Python no longer encodes bucket-layout policy while still validating caller-visible invariants.

## Boundary Ownership

Python owns:

1. user-facing model ID input
2. cache root configuration
3. path-safety validation on returned paths
4. exception mapping into Python-visible errors
5. `_ensure_safetensors(...)` only after a committed Orbax-backed result

Native Rust+Mojo owns:

1. remote artifact discovery
2. model-family-specific artifact selection
3. tokenizer selection policy
4. atomic staging and commit
5. structured success/failure reporting

Python must not:

1. list remote objects
2. infer tokenizer paths from bucket layout
3. rescan the committed tree to reconstruct downloader decisions
4. download individual artifacts itself

## Request Fields: Python -> Native

The Python layer should pass a single structured request with these fields:

1. `model_id`
   - normalized model identifier after `HubManager._clean_model_id(...)`
   - example: `google/gemma-3-4b-it` -> `gemma-3-4b-it`
2. `revision`
   - requested revision or `None`
   - carried through even if the first cut only supports the default revision
3. `cache_root`
   - absolute configured cache root from `HubManager.cache_path`
4. `target_dir_name`
   - normalized final directory name derived from the cleaned model ID
   - keeps Python-side cache layout stable without exposing full bucket paths
5. `require_tokenizer`
   - boolean reflecting whether the caller-visible Python model API requires a tokenizer for this family
   - this is a consumer invariant, not a remote-path hint
6. `allow_orbax`
   - boolean permitting Orbax-backed checkpoints that may require post-commit safetensors conversion

Optional future-safe fields:

1. `expected_family`
   - explicit family tag if later needed for validation or telemetry
2. `request_id`
   - correlation ID for tracing across Python and native logs

## Response Fields: Native -> Python

On success, the native layer should return a structured payload with these required fields:

1. `cache_dir`
   - absolute path to the committed final model directory
2. `committed`
   - boolean proving atomic commit completed
3. `orbax_backed`
   - boolean indicating whether `_ensure_safetensors(...)` must run
4. `tokenizer_required`
   - boolean stating whether this model family requires `tokenizer.model`
5. `tokenizer_relpath`
   - relative path to the tokenizer inside `cache_dir` when required
   - canonical committed location should remain `tokenizer.model`
6. `artifact_summary`
   - compact manifest summary, not a full bucket listing
   - must at least state whether model weights and config metadata were materialized

On failure, the native layer should return structured error fields consumed by Python exception mapping:

1. `error_category`
   - one of `NOT_FOUND`, `TRANSPORT`, `LOCAL_IO`, `INTEGRITY`, `INVALID_PATH`
2. `error_detail`
   - short deterministic detail string suitable for inclusion in a Python exception message
3. `cleanup_state`
   - whether staging or committed output was already removed by the native layer

## Python Validation Invariants

Python must validate only these invariants after the native call:

1. `committed` is `true`
2. `cache_dir` resolves under `self.cache_path`
3. `cache_dir.name` matches the normalized target directory semantics expected by the current cache layout
4. required response fields are present and type-correct
5. if `tokenizer_required` is `true`, `tokenizer_relpath` is present and the file exists after commit
6. if `orbax_backed` is `true`, `_ensure_safetensors(...)` runs before the directory is returned to callers

Python must not validate by rescanning the full artifact tree. It only validates the committed directory, required tokenizer presence, and post-commit conversion outcome.

## Tokenizer Semantics

The tokenizer contract is:

1. Native code decides which tokenizer artifact belongs to the model family
2. Python does not encode bucket paths like `tokenizers/tokenizer_gemma3.model`
3. The committed Python-facing filename stays `tokenizer.model`
4. If `require_tokenizer=true` and the tokenizer is missing, Python raises integrity failure
5. If a family intentionally does not require a tokenizer, the native response must say so explicitly with `tokenizer_required=false`

## Artifact Semantics

The artifact contract is:

1. Native code chooses the remote objects needed for a valid model cache
2. Python receives only a committed cache directory plus compact summary flags
3. The committed cache layout remains compatible with existing consumers in [model.py](/home/cody/code/c/mogemma/src/py/mogemma/model.py)
4. Python does not learn or expose bucket object names as part of the public contract

Required committed outcome:

1. model weights are present in a form acceptable to downstream loaders
2. metadata/config required by the loader is present
3. `tokenizer.model` is present when required
4. the directory is safe to hand directly to `resolve_model(...)` callers after any required safetensors conversion

## Post-Commit Conversion Rule

If `orbax_backed=true`:

1. Python may call `_ensure_safetensors(cache_dir)` after path validation
2. any conversion failure is treated as integrity failure
3. the committed directory must be deleted before surfacing that failure

If `orbax_backed=false`:

1. Python returns `cache_dir` directly after validation
2. no additional artifact discovery or conversion occurs

## Minimal End-to-End Flow

1. Python normalizes `model_id`
2. Python sends one structured request to native downloader
3. Native selects checkpoint and tokenizer artifacts, stages them, and commits atomically
4. Native returns structured success payload
5. Python validates path and required fields
6. Python runs `_ensure_safetensors(...)` only when `orbax_backed=true`
7. Python returns committed `cache_dir`

## Non-Goals

This checkpoint does not require:

1. exposing a per-artifact manifest to Python
2. preserving old Python helper methods for tokenizer-path inference
3. supporting a dual-path `obstore` fallback
4. teaching downstream model classes about remote artifact structure
