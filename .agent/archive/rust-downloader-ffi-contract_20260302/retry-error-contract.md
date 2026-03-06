# Retry/Error Contract (`mogemma-h4n.2.2`)

## Purpose

Define deterministic retry and error propagation rules across Rust downloader, Mojo boundary, and Python caller layers.

## Error Taxonomy

Stable categories:
1. `NOT_FOUND`
2. `AUTH`
3. `NETWORK_TRANSIENT`
4. `NETWORK_PERMANENT`
5. `INTEGRITY`
6. `TIMEOUT`
7. `CANCELLED`
8. `INTERNAL`

Each error payload must include:
1. `category`
2. `message`
3. `retryable: bool`
4. `attempts: int`
5. `artifact_name: str | None`
6. `source_layer: str` (`rust`, `mojo`, or `python`)

## Retry Policy (Choice `B`)

1. Retries are only allowed for `NETWORK_TRANSIENT` and selected `TIMEOUT` errors.
2. Retry budget is bounded by `max_attempts` (default `3`).
3. Backoff strategy is exponential with jitter:
1. `attempt 1`: immediate
2. `attempt 2`: base delay
3. `attempt 3+`: exponential growth capped by max delay
4. `NOT_FOUND`, `AUTH`, `NETWORK_PERMANENT`, `INTEGRITY`, and `INTERNAL` are terminal (no retry).
5. Cancellation is terminal and must stop all in-flight transfers.

## Boundary Mapping Rules

1. Rust emits stable error category and retryable bit.
2. Mojo boundary forwards category unchanged and normalizes message format.
3. Python maps category to deterministic exception classes:
1. `NOT_FOUND` -> `FileNotFoundError`
2. `AUTH` -> `PermissionError`
3. `NETWORK_TRANSIENT` -> `ConnectionError`
4. `NETWORK_PERMANENT` -> `ConnectionError`
5. `INTEGRITY` -> `ValueError`
6. `TIMEOUT` -> `TimeoutError`
7. `CANCELLED` -> `RuntimeError`
8. `INTERNAL` -> `RuntimeError`
4. Exception messages include category token and model id for diagnostics.

## Deterministic Behavior Requirements

1. No silent fallback to alternate downloader path.
2. Final surfaced error must be from terminal category after retries are exhausted.
3. Retry attempts are observable in diagnostics for debugging and CI assertions.
4. Partial artifacts from failed terminal outcomes must not be committed as valid cache entries.

## Chapter 3 Handoff

1. `HubManager.download()` treats retry logic as upstream boundary behavior, not local ad hoc policy.
2. Chapter 3 failure semantics (`mogemma-h4n.3.2`) must be derived from this taxonomy and mapping table.
