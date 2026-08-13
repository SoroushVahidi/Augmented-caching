# KBS Revision Upload Checklist

Manuscript: **KNOSYS-D-26-07461**  
Author: **Soroush Vahidi** (sole author)  
Package folder: `submission_kbs_revision_final/`  
Prepared: 2026-06-21; second-revision update: 2026-08-13

This package contains the files needed for the KBS revision upload to Editorial Manager. Download this folder (or `submission_kbs_revision_final.zip`) from GitHub and upload each item as requested by the revision email.

---

## Files ready for upload

| File | Purpose | Format |
|------|---------|--------|
| `01_Revised_Manuscript.pdf` | Revised manuscript | PDF |
| `02_Response_to_Reviewers.docx` | Point-by-point response / list of changes | DOCX |
| `02_Response_to_Reviewers.md` | Same content (Markdown source) | MD |
| `03_Cover_Letter.docx` | Revised cover letter | DOCX |
| `03_Cover_Letter.md` | Same content (Markdown source) | MD |
| `04_Highlights.docx` | Journal highlights (3–5 bullets) | DOCX |
| `04_Highlights.md` | Same content (Markdown source) | MD |
| `05_CRediT_Author_Statement.docx` | CRediT roles for sole author | DOCX |
| `05_CRediT_Author_Statement.md` | Same content (Markdown source) | MD |
| `06_Declaration_of_Interest.docx` | Competing-interest declaration | DOCX |
| `06_Declaration_of_Interest.md` | Same content (Markdown source) | MD |
| `07_LaTeX_Source/` | LaTeX source (`main.tex`, `refs.bib`, `elsarticle.cls`, `elsarticle-num.bst`, `figures/`) | LaTeX |
| `08_Figures/` | Final PNG figures used in the manuscript | PNG |
| `07_AUTHOR_AGREEMENT_PLACEHOLDER_README.md` | Instructions for official author agreement | MD |

---

## Manual checks before upload

- [ ] **Visually inspect** `01_Revised_Manuscript.pdf` (page count, figures, tables, references, title, author block).
- [ ] **Confirm manuscript number** KNOSYS-D-26-07461 on cover letter and response letter.
- [ ] **Download and complete the official Elsevier/KBS author agreement** from Editorial Manager — see `07_AUTHOR_AGREEMENT_PLACEHOLDER_README.md`. No official completed form is included in this package.
- [ ] **Verify figure quality** in the PDF and in `08_Figures/` (PNG sources are high resolution: ~3150–3400 px wide; suitable for print at typical column widths).
- [ ] **Confirm Editorial Manager upload slots** — some journals request DOCX and LaTeX source separately; upload `.docx` items where DOCX is requested and `07_LaTeX_Source/` where LaTeX source is requested.
- [ ] **Re-read highlights** — each bullet is ≤85 characters where possible; none claim superiority over baselines.

---

## What this revision honestly reports (updated 2026-08-13, second-revision round)

- End-to-end online replay at **capacities 32, 64, and 128 only** (no cap256), under two protocols: the original single-split evaluation (now disclosed as relying on a model with train/test overlap) and a corrected, leakage-free leave-one-family-out evaluation (§3.6).
- **Negative end-to-end result, now against seven baselines including LRB and 3L-Cache:** `evict_value_v1` does not beat LRU, SIEVE, FIFO-Reinsertion, LRB, 3L-Cache, CACHEUS, or HALP under a matched evaluation protocol (§3.6, Table 7).
- **Objective ablation (new):** eviction-loss is the worst or tied-worst of four tested finite-horizon supervision objectives (§3.7, Table 8).
- **Mechanistic diagnosis (new):** target degeneracy and horizon truncation, not model-fitting failure or insufficient data, are the dominant explanation (§3.8).
- **HALP:** now an empirical comparison (independent reimplementation, disclosed fidelity caveat), not analytical-only.
- **Continuation-mismatch causal ablation (new, addresses Reviewer #3):** partially supported, regime-dependent (§3.9); a companion DAgger-style distribution-shift correction is a negative result.
- **Fallback/guard:** remains demoted as unvalidated; no fallback ablation.
- **Overhead:** replaced the local/tmux single-run benchmark with a controlled, repeated-measurement (5-rep) Wulver campaign for LRU/FIFO-Reinsertion/SIEVE/HALP; `evict_value_v1`'s own runtime remains a separate single-run measurement (§3.10).
- **Practical significance (new section, §3.11):** explicitly states no current deployment scenario is justified; reframes contribution as diagnostic/methodological.
- **Solo-author validation** limitation disclosed.

No new experiments were run to prepare this upload package; all second-revision evidence was gathered and validated in prior work on this branch and integrated here.

---

## What is NOT in this package (by design)

- Logs, virtual environments, trained models, raw/processed/derived data
- Canonical merged `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` (does not exist)
- cap256 artifacts
- Draft/skeleton files, `_DRAFT` figures, or scratch reports
- Unofficial author agreement (use Editorial Manager official form)

---

## Suggested upload order

1. Revised manuscript PDF  
2. Response to reviewers DOCX  
3. Cover letter DOCX  
4. Highlights DOCX  
5. CRediT author statement DOCX  
6. Official author agreement (from Editorial Manager — complete separately)  
7. Declaration of interest DOCX  
8. LaTeX source folder (if requested)  
9. Figure PNGs (if requested separately from LaTeX)

---

## Package integrity

After download from GitHub, verify:

```bash
find submission_kbs_revision_final -maxdepth 3 -type f | sort
```

Expected: PDF, five DOCX pairs (MD + DOCX), LaTeX source tree, three PNG figures, placeholder README, and this checklist. No `.venv`, logs, or data directories.
