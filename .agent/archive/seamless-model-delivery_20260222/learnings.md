# Learnings: Seamless Model Delivery

## [2026-02-22] - Zero-config downloads

- **Implemented:** Migrated huggingface-hub to base deps. Hardened local > cache > remote resolution.
- **Learnings:**
  - Move optional deps to base when they enable critical user workflows.
  - Clear resolution hierarchy prevents ambiguous fallbacks.
  - Optional dependency facades become dead code after policy changes — audit.
  - Cache directory behavior must be stable and documented.
