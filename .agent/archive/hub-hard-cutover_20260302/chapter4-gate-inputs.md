# Chapter 4 Gate Inputs

## Objective

Define the minimum benchmark coverage, failure-injection coverage, and release-blocking signals required before the `HubManager.download()` hard cutover to the Rust+Mojo CFFI downloader can ship.

## Benchmark Matrix

The benchmark matrix should cover both performance and contract preservation.

### Core scenarios

1. Warm-cache resolve for a previously committed model directory
   - entrypoint: `HubManager.resolve_model(..., download_if_missing=True, strict=True)`
   - expected behavior: zero network/native downloader work
   - release signal: remains a local fast-path check
2. Cold-path download for a standard text-generation model
   - representative model family: `gemma-3-4b-it`
   - measures: total wall time, native download time, Python overhead before/after native call
3. Cold-path download for a Nano-family model
   - representative family: `gemma3n`
   - measures: artifact selection correctness and tokenizer handoff in addition to timing
4. Cold-path download for an Orbax-backed checkpoint requiring `_ensure_safetensors(...)`
   - measures: download time, post-commit conversion time, cleanup correctness on conversion failure
5. Repeat download of an already cached model after one successful cold-path run
   - measures: no false redownload; warm-cache latency remains low

### Mandatory benchmark metrics

1. end-to-end `resolve_model(...)` latency
2. native downloader wall time
3. Python orchestration overhead
4. final committed cache size
5. `_ensure_safetensors(...)` conversion overhead when applicable
6. peak temporary disk footprint during staging for large-model cold path

## Reliability and Fault-Injection Matrix

The release process should inject failures at every contract boundary.

### Upstream and transport failures

1. model not found
2. DNS/connect timeout
3. retry exhaustion
4. interrupted transfer before commit
5. truncated remote payload

Expected outcome:

1. deterministic exception mapping
2. no final cache tree
3. no warm-cache false positive on retry

### Local filesystem failures

1. cache-root directory creation denied
2. write failure during staging
3. disk-full during staging
4. atomic rename failure at commit
5. cleanup failure after native abort

Expected outcome:

1. local failure remains distinguishable from transport failure
2. final cache tree never appears as valid unless commit completed cleanly

### Integrity and contract failures

1. checksum mismatch
2. format mismatch in downloaded artifact set
3. missing required tokenizer for tokenizer-required families
4. invalid returned cache path outside configured cache root
5. malformed native success payload missing required fields
6. `_ensure_safetensors(...)` failure after native commit

Expected outcome:

1. fail closed
2. delete committed output if integrity break is discovered after commit
3. never return a path that downstream callers can treat as valid

## Release-Blocking Signals

The cutover should be blocked from release if any of the following are true:

1. warm-cache resolve performs native downloader work
2. any failure mode leaves a final cache directory that passes `_has_model_files(...)`
3. `NOT_FOUND` no longer maps to `HubManager.ModelNotFoundError`
4. local filesystem failures are remapped to transport-style failures
5. invalid returned paths can escape `self.cache_path`
6. `_ensure_safetensors(...)` can leave a partially converted cache tree visible as valid
7. Python-side orchestration reintroduces per-artifact transfer logic or threadpool scheduling

## Suggested Quantitative Gates

These are the minimum quantitative checks needed before calling the cutover ready:

1. Warm-cache `resolve_model(...)` latency must stay within a small constant-factor range of the current local-cache baseline
2. Python orchestration overhead must remain a minority of cold-path wall time; it should not dominate the native downloader work
3. Cold-path retry after an interrupted transfer must succeed without manual cache cleanup
4. Every injected failure class must leave zero false-positive cache hits on immediate rerun

The exact numeric thresholds should be finalized once the native downloader prototype exists, but the implementation PR must publish measured before/after baselines for each benchmark scenario above.

## Evidence Required for Closure

1. benchmark output for each required scenario
2. failure-injection results for each required failure class
3. explicit assertion list showing which tests enforce each release blocker
4. confirmation that Python no longer contains `obstore` or threadpool-based download logic

## Deferred Safetensors Scope

The initial hard cutover should defer any safetensors work that is not required to preserve current caller-visible behavior.

Keep in initial scope:

1. `_ensure_safetensors(...)` still runs for Orbax-backed committed results because existing callers expect a usable local model directory
2. cleanup on conversion failure remains a release blocker

Defer out of initial scope:

1. optimizing conversion throughput
2. changing the on-disk safetensors layout
3. parallelizing or offloading conversion beyond what current behavior already requires
4. introducing new user-facing configuration knobs for conversion policy

## Implementation Notes

1. Benchmark instrumentation should isolate native download time from Python orchestration time so regressions can be attributed cleanly
2. Reliability gates should be written against caller-visible invariants, not internal implementation details
3. The first implementation PR should treat any unresolved gate as a blocker, not as a follow-up polish item
