# KBS Revision Upload Checklist

> **Status (2026-08-14).** This checklist was prepared 2026-06-21 and is
> **not** a current scientific summary. Several bullets below (HALP
> “analytical only”, fallback framing, page/experiment scope) are
> superseded. For current reviewer evidence use
> `01_Revised_Manuscript.pdf`, `02_Response_to_Reviewers.md`,
> `07_LaTeX_Source/`, and `../docs/reviewer/START_HERE.md`.

Manuscript: **KNOSYS-D-26-07461**  
Author: **Soroush Vahidi** (sole author)  
Package folder: `submission_kbs_revision_final/`  
Prepared: 2026-06-21 (historical packaging note)

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

## What this revision honestly reports

- End-to-end online replay at **capacities 32, 64, and 128 only** (no cap256).
- **Negative end-to-end result:** `evict_value_v1` does not beat LRU, SIEVE, or FIFO-Reinsertion at any evaluated capacity.
- **HALP:** analytical differentiation only; no empirical HALP comparison.
- **Fallback/guard:** demoted as unvalidated; no fallback ablation.
- **Overhead:** includes a local/tmux wall-clock benchmark (not Wulver/Slurm).
- **Solo-author validation** limitation disclosed.

No new experiments were run to prepare this upload package.

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
