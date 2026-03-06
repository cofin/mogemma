# Knowledge Base

> Persistent learnings from all completed flows.
> For actionable patterns, see [patterns.md](../patterns.md).
> Drill into specific entries only when relevant to current work.

## Entries

| Flow ID | Completed | Topics | Summary |
|---------|-----------|--------|---------|
| launch-cleanup | 2026-02-22 | Cleanup | Dead code removal, vision code purge, fail-fast over fallbacks. |
| ci-setup | 2026-02-15 | CI/CD | GitHub Actions lint/type-check/test matrix, Mojo index config. |
| release-setup | 2026-02-15 | CI/CD, Packaging | bump-my-version, trusted publishing, platform-specific wheels. |
| project-readme | 2026-02-15 | Docs | README with quick start, optional extras, API surface docs. |
| core-inference-parity | 2026-02-22 | Inference | Contract-first Mojo core replacement, placeholder removal. |
| python-runtime-contracts | 2026-02-22 | Inference, API | Sync/async contract hardening, error taxonomy, deterministic sampling. |
| seamless-model-delivery | 2026-02-22 | Model, DX | Zero-config downloads, local>cache>remote resolution hierarchy. |
| quality-gates | 2026-02-22 | Quality | Hard-fail lint gates, suppression audit, CI-local parity. |
| distribution-release-ops | 2026-02-22 | Packaging | Wheel/sdist validation, release runbook, go/no-go checklist. |
| docs-claims-validation | 2026-02-22 | Docs | Claims-to-evidence mapping, marketing language removal. |
| performance-baselines | 2026-02-22 | Performance | Reproducible benchmarks, regression thresholds, baseline capture. |
| pure-mojo-bridge | 2026-02-25 | Architecture | Dependency purge, zero-copy FFI, SafetensorsLoader. |
| pure-mojo-primitives | 2026-02-25 | Architecture | SIMD RMSNorm, GeGLU, RoPE, matmul implementations. |
| pure-mojo-transformer | 2026-02-25 | Architecture | GQA, MLP, decoder layer, embedding pooling. |
| pure-mojo-generation | 2026-02-25 | Architecture | KV-Cache, incremental decoding, stateful FFI. |
| gemma3-nano-conversion | 2026-02-25 | Model | Nano 671-tensor Orbax conversion, model family detection. |
| gemma3-nano-inference | 2026-02-23 | Model | AltUp, Laurel, per-layer embeddings, architecture dispatch. |
| core-embedding-padding | 2026-02-24 | Inference | Jagged array padding migration to Mojo backend. |
| validation-enhancements | 2026-02-25 | Quality | Semantic prompt gates, deterministic validation, failure taxonomy. |
| embedding-model-id-policy | 2026-02-25 | Inference, API | Removed invalid default IDs, actionable ModelNotFoundError. |
| cleanup-and-migrate | 2026-02-25 | Build | hatch-mojo plugin migration, declarative build config. |
| api-max-tokens | 2026-02-25 | API | Breaking rename max_new_tokens -> max_tokens. |
| mojo-namespace-refactor | 2026-02-23 | Architecture | Namespace restructuring, layout test validation. |
| pypi-cibuildwheel | 2026-02-25 | Packaging | GLIBCXX mismatch blocks manylinux wheels for Mojo. |
| advanced-dx-distribution | — | DX | Async bridging, local-first hub, rich-click CLI. |
| text-inference-engine | — | Inference | Bridge overhead benchmarking (sub-millisecond). |
| vision-language-bridge | — | Multimodal | Interleaved multimodal inputs API design. |
| core-generation-fix | Active | Inference, Model | IT turn formatting, Nano KV-sharing, AltUp, activation sparsity. |

## Topic Index

- **Architecture** — pure-mojo-bridge, pure-mojo-primitives, pure-mojo-transformer, pure-mojo-generation, mojo-namespace-refactor
- **API** — python-runtime-contracts, api-max-tokens, embedding-model-id-policy
- **Build** — cleanup-and-migrate
- **CI/CD** — ci-setup, release-setup
- **Cleanup** — launch-cleanup
- **Docs** — project-readme, docs-claims-validation
- **DX** — seamless-model-delivery, advanced-dx-distribution
- **Inference** — core-inference-parity, core-embedding-padding, text-inference-engine, core-generation-fix, embedding-model-id-policy
- **Model** — gemma3-nano-conversion, gemma3-nano-inference
- **Multimodal** — vision-language-bridge
- **Packaging** — distribution-release-ops, pypi-cibuildwheel
- **Performance** — performance-baselines
- **Quality** — quality-gates, validation-enhancements
