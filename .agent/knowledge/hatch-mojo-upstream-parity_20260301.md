# Knowledge: hatch-mojo-upstream-parity_20260301

**Completed:** 2026-03-06
**Archived:** 2026-03-06

## Topics
hatch-mojo, upstream parity, pyproject.toml, workaround audit, packaging, drift risk

## Learnings
- Workaround comments in `pyproject.toml` can outlive upstream fixes; parity audits should pair comments with upstream release evidence before retaining overrides.
- Toggle-in-CI patterns (for example `sed` editing project config during build) increase drift risk and should be replaced with a canonical source-of-truth policy when upstream behavior stabilizes.
- The Linux GLIBCXX/toolchain issue and hatch-mojo runtime-linking issue are related operationally but should be tracked as separate constraints to avoid stale assumptions.

## Summary
Audited active mogemma hatch-mojo workarounds against hatch-mojo v0.1.5 to produce an evidence-backed keep/remove matrix. Identified that CI toggle patterns and stale workaround comments create drift risk, and that GLIBCXX and runtime-linking issues should be tracked independently.
