# Repository cleanup report (journal-readiness navigation)

**Date:** 2026-04-12 (approximate; see git history for exact commit).  
**Scope:** Documentation and navigation only—**no** experiment logic changes, **no** deletion of `analysis/` or other provenance artifacts, **no** script moves (would break imports, docs, and muscle memory).

---

## 1. Organizational audit (before changes)

### What was working well

- Clear **canonical `heavy_r1`** filenames were already documented in `docs/evict_value_v1_kbs_canonical_artifacts.md` and `analysis/README.md`.
- `docs/kbs_manuscript_workflow.md` already separated canonical vs exploratory.
- `scripts/README.md` already described directory families without requiring physical moves.

### What was confusing for external readers

1. **No single top-level “submission checklist”** — readers had to assemble workflow + artifacts + submission index from multiple entry points.
2. **`docs/` had no index** — 50+ markdown files (KBS, pairwise, internal, hygiene) without a grouped TOC.
3. **README was long** — strong content but buried the KBS path under a large policy catalog.
4. **`analysis/`** — easy to confuse `*_heavy_r1` with unsuffixed or `heavy_smoke` files; benefit from an even more explicit warning at the top of `analysis/README.md`.
5. **`tables/manuscript/` and `figures/manuscript/`** — no short README explaining generated vs hand-edited and dependency on canonical CSV.
6. **`reports/`** — no README explaining manifest vs audits.

### Intentionally **not** changed

- **No file moves or renames** under `scripts/`, `analysis/`, `src/`, or `slurm/` (preserves paths in papers, Slurm, CI, and git history).
- **No removal** of legacy root-level `analysis/*.csv` or exploratory campaign trees.
- **No edits** to training/eval/simulator code or experiment defaults.
- **No merging** of `kbs_manuscript_workflow.md` into a single mega-doc (kept as narrative primary; checklist added separately).

---

## 2. What was added or updated

| Change | Rationale |
|--------|-----------|
| **`CANONICAL_KBS_SUBMISSION.md` (repo root)** | Single obvious entry for reviewers and new authors |
| **`docs/README.md`** | Grouped index; points to checklist as primary for KBS |
| **`docs/repository_cleanup_report.md`** (this file) | Audit trail for navigation edits |
| **`README.md`** | Shorter top: purpose, main method, baselines, KBS path, reproduce, exploratory; links to new checklist and `docs/README.md` |
| **`docs/kbs_manuscript_workflow.md`**, **`docs/kbs_manuscript_submission_index.md`**, **`docs/repo_map.md`**, **`docs/reproducibility_and_artifacts.md`** | One-line pointers to `CANONICAL_KBS_SUBMISSION.md` / `docs/README.md` |
| **`analysis/README.md`** | Stronger canonical vs non-canonical warning; taxonomy table; `heavy_smoke` callout |
| **`scripts/README.md`** | “Layout stability” note: flat layout by design |
| **`tables/manuscript/README.md`**, **`figures/manuscript/README.md`**, **`reports/README.md`** | Short generated-output orientation |
| **`slurm/README.md`**, **`tests/README.md`** | Pointers to canonical batch files and pytest |

---

## 3. Repository navigation guide (summary)

1. **New to the repo:** Read root **`README.md`**, then **`CANONICAL_KBS_SUBMISSION.md`** if you care about the KBS paper path.
2. **All documentation:** **`docs/README.md`** (grouped index).
3. **Deep workflow:** **`docs/kbs_manuscript_workflow.md`** + **`docs/wulver_heavy_evict_value_experiment.md`**.
4. **Analysis outputs:** **`analysis/README.md`** — always check for **`_heavy_r1`** suffix before citing KBS main numbers.
5. **Scripts:** **`scripts/README.md`** — paths stay at documented locations.
6. **Generated LaTeX bundle:** `tables/manuscript/`, `figures/manuscript/`, `reports/manuscript_artifacts/` (see per-directory READMEs).

---

## 4. Follow-ups (optional, not done here)

- Periodic grep for broken internal links after large doc moves (none performed in this pass).
- If a future maintainer wants physical `scripts/` subpackages, plan a dedicated migration with CI and doc grep—not part of this conservative pass.
