# DOCX submission package status (2026-06-19)

Read-only inspection. No files created/converted; no manuscript finalized.

## What was checked

```bash
find . -iname "*submission_kbs_revision_docx*" -not -path "*/.git/*"
find / -maxdepth 4 -iname "*submission_kbs_revision_docx*"
find . -iname "*.docx" -not -path "*/.git/*" -not -path "*/.venv*/*"
unzip -l Decision_aligned_eviction_value_prediction_for_robust_learning_augmented_caching.zip
find . -iname "*highlight*" -o -iname "*credit*" -not -path "*/.git/*"
find . -iname "*cover*letter*" -not -path "*/.git/*" -not -path "*/.venv*/*"
find . -iname "*declaration*" -not -path "*/.git/*" -not -path "*/.venv*/*"
```

## Findings

- **No `submission_kbs_revision_docx` directory exists anywhere** — neither
  in this repo (any path, any branch) nor as a sibling directory on the
  filesystem. This matches the prior completion audit's finding
  (`reports/kbs_revision_completion_audit.md`, "DOCX skeleton directory:
  NOT FOUND") and is reconfirmed directly in this pass.
- **Zero `.docx` files exist anywhere in the repository.**
- The only real submission package on disk is
  `Decision_aligned_eviction_value_prediction_for_robust_learning_augmented_caching.zip`
  (repo root, added in commit `e9ac132`), and it is **entirely LaTeX**:

  | File | Role |
  |---|---|
  | `main.tex` | Manuscript body (~8,919 words per prior audit) |
  | `refs.bib` | Bibliography (37 entries) |
  | `cover-letter.tex` | Cover letter — **written for an initial submission**, not this revision (addressed "Please consider the enclosed manuscript... for publication") |
  | `author-agreement.tex` | Author agreement |
  | `elsarticle.cls`, `elsarticle-num.bst`, `elsarticle-num-names.bst`, `elsarticle-harv.bst` | Elsevier LaTeX template/style files |
  | `figures/method_overview.png`, `figures/figure4_ablation.png` | Two of the manuscript's figures (others are generated separately under `figures/manuscript/`, not bundled in this zip) |

  10 files total, no DOCX, no PDF.

- **No Highlights file or section, in any format.** Confirmed by direct
  `find` and (per the prior completion audit) by grep across all `.tex`
  files in the zip.
- **No standalone CRediT author-contribution statement, in any format.**
- **No standalone Declaration-of-Interest file.** A "Declaration of
  Competing Interest" section exists, but only **inline inside `main.tex`**
  — not as a separate file in any format. This may or may not satisfy
  Editorial Manager's requirement, which often (for Elsevier journals)
  expects this as a distinct uploaded item separate from the manuscript
  body.

## Direct answers to the task's checklist

| Question | Answer |
|---|---|
| Which Word files exist? | **None. Zero `.docx` files exist anywhere in the repository.** |
| Which are missing? | All of them: revised manuscript (DOCX or otherwise up to date — see below), response-to-reviewers letter, cover letter (revision-specific), Highlights, CRediT statement, standalone Declaration of Interest. |
| Is the revised manuscript DOCX pending? | Yes — pending in the strongest sense: not just "not yet converted to DOCX," but the underlying LaTeX manuscript itself is **not yet revision-ready** (zero online-replay/Table 3 results; stale offline-ablation table per `kbs_stale_artifact_refresh_plan.md`; SIEVE/FIFO-Reinsertion baseline gaps per `kbs_baseline_gap_action_plan.md`). Converting to DOCX before the content itself is finalized would just create a second stale artifact to maintain. |
| Is LaTeX-only source insufficient for final submission? | **Depends on the actual KNOSYS/Editorial Manager requirement, which is not derivable from anything in this repo.** Two plausible scenarios, both common for Elsevier journals, with very different effort implications: (a) the main manuscript must be submitted as Word/DOCX (would require full LaTeX→DOCX conversion — non-trivial with equations, multi-row tables, and a 37-entry bibliography; conversion tools like `pandoc` exist but typically need manual cleanup for `elsarticle`-class documents); or (b) only certain ancillary files (cover letter, response-to-reviewers letter, possibly Highlights) need to be Word/DOCX while the manuscript itself stays LaTeX→PDF, which is the more common pattern for revision submissions to KBS specifically. **This cannot be resolved without checking the actual KNOSYS-D-26-07461 Editorial Manager submission instructions or notification email.** (Note, 2026-06-19: the verbatim reviewer comments, previously cited here as an analogous external-information blocker, are no longer missing — they were supplied by the user and are recorded in `reports/kbs_real_reviewer_comments.md`. The DOCX-format requirement itself is a separate, still-unresolved external-information question.) |
| What can be created now vs. after canonical results? | **Now, independent of canonical results**: a revision-specific cover letter (rewriting `cover-letter.tex`'s initial-submission framing), a CRediT statement (single-author, mechanical), Highlights (format/length still needs the journal's actual requirement, but drafting candidate bullet points doesn't need new results). **Must wait for canonical results (cap64/128/256 + merge + stale-artifact refresh)**: the revised manuscript itself, since its main quantitative claims don't exist yet — finalizing any manuscript file (LaTeX, PDF, or DOCX) before that would need a second revision pass once cap64+ lands. |

## Explicit non-action (original inspection pass, 2026-06-19)

Per instruction, **the revised manuscript DOCX (or any other DOCX) was not
created, converted, or finalized in this pass.** This report is inspection
and classification only.

---

## Update: submission-package skeletons created (2026-06-19, while
`cap64_with_sieve_fifo` runs)

A follow-on pass, building directly on this report and on
`reports/kbs_docx_package_action_plan.md`'s §2/§5 findings (items 2-7 are
draftable now with low conversion-fidelity risk since they are prose-only,
unlike the equation/table/algorithm-heavy main manuscript), created actual
draft skeleton files for the **6 non-manuscript** submission-package items
in `submission_kbs_revision_docx/`. The revised manuscript DOCX (item 1)
remains explicitly **not created**, per instruction and per the action
plan's own blocking rationale (content not yet revision-ready pending
cap64/128/256).

### Files created

| File (markdown source) | Corresponding `.docx` | Status |
|---|---|---|
| `response_to_reviewers_skeleton.md` | `response_to_reviewers_skeleton.docx` | Draft skeleton — carries 16 `[INSERT FINAL CANONICAL RESULT AFTER CAP64/FINAL SWEEP]` placeholders, all confirmed surviving the `.docx` round-trip |
| `cover_letter_skeleton.md` | `cover_letter_skeleton.docx` | Draft skeleton — 2 placeholders (contribution/result summary paragraph, carry-forward paragraph) |
| `highlights_skeleton.md` | `highlights_skeleton.docx` | Draft skeleton — 5 candidate bullets, 1 conditional placeholder |
| `credit_author_statement_skeleton.md` | `credit_author_statement_skeleton.docx` | Draft, no result-dependent placeholder — mechanical, single-author statement |
| `declaration_of_interest_skeleton.md` | `declaration_of_interest_skeleton.docx` | Draft, no result-dependent placeholder — verbatim text lifted from `manuscript_source/main.tex` lines 579-581 |
| `author_agreement_placeholder.md` | `author_agreement_placeholder.docx` | Placeholder, no result-dependent content — verbatim text lifted from `manuscript_source/author-agreement.tex` |

Each markdown source uses the real AE/R2-MC1-3/R3-Issue1-7/R3-Minor8-9/
R3-Rec1-8 labels (per `reports/kbs_real_reviewer_comments.md`) where
applicable (the response-to-reviewers skeleton only). Every `.docx` was
generated via `pandoc 3.9.0.2` (`/home/soroush/bin/pandoc`), each exit 0.
Because these 6 documents are short and prose-only (no equations, no
`algorithmicx` blocks, no `\ref{...}` cross-references, no tables), they
do not trigger any of the four fidelity failure modes documented in
`reports/kbs_docx_package_action_plan.md` §3 for the main manuscript —
confirmed directly by round-tripping `response_to_reviewers_skeleton.docx`
and `cover_letter_skeleton.docx` back to plain text and diffing against
the source: all placeholder text and paragraph content survives, only
line-wrapping differs.

### What this update does NOT do

- Does not create, convert, or finalize the **revised manuscript DOCX**
  (item 1) — still blocked on cap64/128/256 and on applying several
  already-drafted text edits to `manuscript_source/main.tex` beyond what
  the separate safe-manuscript-source-edits pass already applied (see
  `reports/kbs_safe_manuscript_source_edits_report.md`).
- Does not treat any of the 6 new files as submission-final. Every one
  retains explicit "DRAFT SKELETON — NOT FINAL" or "PLACEHOLDER" status
  banners and, where applicable, load-bearing
  `[INSERT FINAL CANONICAL RESULT AFTER CAP64/FINAL SWEEP]` placeholders
  that must not be guessed at.
- Does not touch, stop, or restart `cap64_with_sieve_fifo` (confirmed
  running, untouched, throughout this pass) and does not launch
  cap128/cap256.
- Does not push, merge, delete, or overwrite any existing artifact —
  `submission_kbs_revision_docx/README.md` and `README_next_steps.md` are
  unmodified; only new files were added alongside them.
