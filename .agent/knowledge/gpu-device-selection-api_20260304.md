# Knowledge: gpu-device-selection-api_20260304

**Completed:** 2026-03-06
**Archived:** 2026-03-06

## Topics
gpu, device-selection, api-design, cpu, gpu:N, fallback-policy, capability-detection

## Learnings
No learnings recorded.

## Summary
Defined and implemented deterministic device selection semantics (cpu, gpu, gpu:N) with explicit capability detection and strict error behavior when requested GPUs are unavailable. Established fallback policy and device-index parsing at the Python API layer with hooks for future runtime capability probing.
