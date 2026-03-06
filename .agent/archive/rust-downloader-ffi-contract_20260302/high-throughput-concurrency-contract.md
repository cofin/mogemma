# High-Throughput Concurrency Contract (`mogemma-h4n.2.4`)

## Purpose

Define the bounded high-throughput download execution model for Rust downloader calls exposed via Mojo CFFI.

## Concurrency Model

1. Unit of parallelism is artifact transfer task.
2. Scheduler uses bounded worker pool with `max_workers = min(request.concurrency, runtime_cap)`.
3. Queue is bounded; producer blocks or rejects when queue limit is reached.
4. Per-artifact chunks can be parallelized only if integrity guarantees are preserved.

## Throughput Controls

1. `request.concurrency` is required and validated (`>=1`).
2. Runtime cap is fixed by implementation and surfaced in diagnostics.
3. Optional per-host connection cap prevents endpoint overload.
4. Small artifacts may be batched to reduce dispatch overhead.

## Backpressure Rules

1. No unbounded in-memory buffering.
2. Queue saturation triggers controlled backpressure, not uncontrolled task spawning.
3. Backpressure signals are surfaced to diagnostics for CI visibility.

## Cancellation and Timeout

1. Cancellation token is propagated Python -> Mojo -> Rust.
2. Cancelling a request stops scheduling new transfers and interrupts active transfer loops.
3. Timeout enforces hard upper bound on overall request duration.
4. Cancelled/timed-out requests return deterministic terminal error categories (`CANCELLED`/`TIMEOUT`).

## Cache Integrity Under Concurrency

1. Writes use temp files and atomic commit semantics.
2. No concurrent writer may commit over another committed artifact version for the same request key.
3. Failed/cancelled workers must clean temporary files for their task scope.
4. Final response includes only successfully committed artifact paths.

## Fairness and Isolation

1. Model requests should not starve smaller transfer sets indefinitely.
2. Worker scheduling avoids head-of-line blocking when possible.
3. If global throttling is applied, per-request fairness policy must be deterministic.

## Observability Requirements

1. Diagnostics include worker count, queue pressure indicators, retries, and total elapsed time.
2. Error payloads include attempt count and affected artifact identity when available.
3. Concurrency-related failures must be distinguishable from network failures.

## Chapter 3 Handoff Constraints

1. `HubManager.download()` performs a single boundary call and relies on this model for transfer orchestration.
2. Python does not implement parallel transfer loops in hard-cutover mode.
3. Chapter 4 benchmarks must validate throughput/latency behavior at low/medium/high concurrency settings.
