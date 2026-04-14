# Learnings: hatch-mojo-upstream-parity_20260301

## 2026-03-02

- Workaround comments in `pyproject.toml` can outlive upstream fixes; parity audits should pair comments with upstream release evidence before retaining overrides.
- Toggle-in-CI patterns (for example `sed` editing project config during build) increase drift risk and should be replaced with a canonical source-of-truth policy when upstream behavior stabilizes.
- The Linux GLIBCXX/toolchain issue and hatch-mojo runtime-linking issue are related operationally but should be tracked as separate constraints to avoid stale assumptions.

