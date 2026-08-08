# DOCX submission package text bank (results-independent, 2026-06-19)

Status: **text bank only — not a finalized DOCX manuscript or package.**
Drafted while `cap32_with_sieve_fifo` runs. Per
`reports/kbs_docx_submission_package_report.md`, the only real submission
package that exists today is entirely LaTeX
(`Decision_aligned_eviction_value_prediction_for_robust_learning_augmented_caching.zip`,
committed 2026-06-17): `main.tex` (~8,919 words), `cover-letter.tex` (written
for the *initial* submission, not this revision), `author-agreement.tex`,
and `.cls`/`.bst` support files. There are **zero `.docx` files anywhere in
the repository**. No Highlights file, no standalone CRediT statement exist
in any format. The existing AI-tool disclosure and Declaration of Competing
Interest live only inline inside `main.tex` (lines ~540, ~553-555), quoted
below for reference and reused/extended rather than contradicted.

Whether KNOSYS actually requires the *manuscript itself* in DOCX, or only
certain ancillary files, is unresolved and not derivable from this repo —
see `reports/kbs_docx_submission_package_report.md`. This bank drafts text
that would work either way (it does not assume the answer).

---

## 1. Revised cover letter (revision-specific)

The existing `cover-letter.tex` is written for an *initial* submission
("Please consider the enclosed manuscript... for publication"). For a
revision, it needs to be reframed around the response to reviewers rather
than a first-time pitch. Draft replacement opening and closing (body
paragraphs 2-3 of the original, describing the contribution, can largely
carry over once the contributions list itself is finalized — see
`title_and_contribution_reframing_options.md`):

> Dear Editor,
>
> Please find enclosed our revised manuscript, entitled
> [INSERT FINAL TITLE — see `title_and_contribution_reframing_options.md`],
> submitted in response to the Associate Editor's and both reviewers'
> comments on manuscript KNOSYS-D-26-07461. We thank the Associate Editor
> and reviewers for their detailed and constructive feedback, which has
> substantially improved this revision.
>
> In summary, we have: (1) added an end-to-end, online cache-replay
> evaluation across multiple cache capacities and trace families, reporting
> results transparently regardless of outcome
> [INSERT FINAL CANONICAL RESULT summary sentence]; (2) added SIEVE and
> FIFO-Reinsertion as additional modern baselines; (3) added a quantitative
> discussion of computational overhead, separating measured offline costs
> from code-verified online complexity; (4) clarified our differentiation
> from HALP and explained why a faithful empirical reproduction was judged
> out of scope; (5) revised our treatment of the guarded fallback mechanism
> to accurately reflect its current (unvalidated) evidentiary status; (6)
> shortened the manuscript and reduced repetition; and (7) added a candid
> discussion of our validation practices as a single author using AI
> coding tools. A detailed, point-by-point response to each reviewer
> comment is enclosed separately.
>
> [Carry forward, once finalized, the contribution-description paragraphs
> from the original cover letter, updated to match whichever title/
> contribution option from `title_and_contribution_reframing_options.md` is
> adopted.]
>
> The manuscript is original, has not been published previously, and is
> not under consideration for publication elsewhere. The manuscript has
> been approved by all authors. As this is a single-author paper, there is
> only one corresponding author: Soroush Vahidi (sv96@njit.edu).
>
> Thank you for your time and consideration in reviewing this revision.
>
> Sincerely,

**Do not finalize** the contribution summary in paragraph 2 until the
canonical sweep result and the resulting title/contribution choice are
settled — the bracketed placeholders above are load-bearing, not
decorative.

---

## 2. Highlights

KNOSYS/Elsevier Highlights are typically 3-5 bullet points, each ≤85
characters. No prior Highlights file exists in any format
(`reports/kbs_docx_submission_package_report.md` confirms this directly).
Draft candidates, written to remain true regardless of the final
sweep outcome (specific result language deferred):

> - Proposes a finite-horizon counterfactual replay target for cache
>   eviction scoring
> - Evaluates the resulting policy end-to-end via online cache replay, not
>   only offline metrics
> - Adds SIEVE and FIFO-Reinsertion as modern baselines, addressing a prior
>   coverage gap
> - Reports per-workload results, including conditions where the learned
>   policy does not outperform LRU
> - Releases full reproducibility artifacts: code, trained models, and
>   evidence logs

[INSERT, IF FINAL RESULTS SUPPORT IT: replace bullet 4 with a more specific
quantitative highlight, e.g. "...improves miss ratio by X% on Y of Z
workloads at capacity C" — only if the canonical sweep actually shows this.
Otherwise keep bullet 4 as written above.]

---

## 3. CRediT author statement

Single-author paper; CRediT statement is mechanical (lists every role
performed by the sole author). No standalone CRediT statement exists today
in any format. Draft:

> **Soroush Vahidi**: Conceptualization, Methodology, Software, Validation,
> Formal analysis, Investigation, Data Curation, Writing – Original Draft,
> Writing – Review & Editing, Visualization, Project Administration.

This can be finalized today — it does not depend on the canonical sweep.

---

## 4. Declaration of (Competing) Interest

Already exists, inline in `main.tex` (line ~553-555), and needs no
content change for this revision unless circumstances changed:

> The author declares that there is no known competing financial interest
> or personal relationship that could have appeared to influence the work
> reported in this paper.

If KNOSYS requires this as a standalone file/section separate from the
manuscript body, this exact text can be lifted verbatim — no rewrite
needed. This can be finalized today.

---

## 5. AI-tool disclosure

Already exists, inline in `main.tex` (line ~540):

> During the preparation of this manuscript, the author used ChatGPT for
> writing assistance and manuscript refinement, and Perplexity AI for
> literature exploration and background research. During the
> implementation stage of the project, the author used Cursor, ChatGPT
> Codex, and GitHub Copilot for coding assistance, and Gemini for limited
> recommendations. All generated suggestions and outputs were reviewed,
> verified, and edited by the author, who takes full responsibility for the
> content of the manuscript, the code, and the reported results.

This existing statement already covers AI-assisted coding generally; it
does not need new tool names added for this revision unless the actual
toolset used during the revision differs (e.g., if this revision's coding
work — SIEVE/FIFO-Reinsertion implementation, evidence-pipeline scripting —
used a different specific assistant, that should be named here for
accuracy rather than left implicit). Suggested append, if accurate for this
revision cycle:

> [INSERT, IF ACCURATE: a sentence naming the specific AI coding assistant
> used during this revision cycle's implementation work (e.g., SIEVE/
> FIFO-Reinsertion policy implementation, evidence-tracking scripts), to
> keep the disclosure current rather than only describing the original
> submission's toolset.]

This connects directly to R3-Issue7 (single-author/AI-tool validation
concern) — see `response_paragraph_bank.md` §9 and
`manuscript_insertions_draft.md` §7 for the corresponding rebuttal-letter
and manuscript-body text. Keep all three consistent: do not claim more
validation rigor in one than another.

---

## Notes

- Nothing in this bank requires waiting for cap64/128/256: items 2-4 are
  fully draftable today and not blocked. Item 1 (cover letter) has one
  load-bearing placeholder (the contribution/result summary) that depends
  on the title/contribution decision and the canonical sweep outcome. Item
  5 needs only a factual check (which AI tools were actually used this
  cycle), not new compute.
- This is a text bank, not a finalized DOCX or LaTeX file. No `.docx` file
  has been created by this pass, and `main.tex`/`cover-letter.tex` inside
  the existing zip have not been modified.
