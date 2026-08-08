# Submission readiness assessment (2026-06-21, final revision audit pass)

Estimates below are derived directly from
`reports/kbs_final_reviewer_coverage_audit.md` (Task 4's 22-row table),
`reports/kbs_manuscript_shortening_execution_plan.md` (Task 1),
`reports/kbs_fallback_final_decision_report.md` (Task 2),
`reports/kbs_halp_positioning_final_audit.md` (Task 3), and a direct
re-check of `submission_kbs_revision_docx/` in this pass. All percentages
are explicitly weighted estimates, not measured quantities — the weighting
rationale is shown so the numbers can be challenged or recomputed.

## 1. Reviewer concerns completed: **~50%**

From the Task 4 table (22 rows): 0 FULLY ADDRESSED, 11 SUBSTANTIALLY
ADDRESSED, 7 PARTIALLY ADDRESSED, 4 NOT ADDRESSED. Scoring
Fully=100%, Substantially=70%, Partially=35%, Not=5%:

```
(0×100 + 11×70 + 7×35 + 4×5) / 22 = (770 + 245 + 20) / 22 = 1035/22 ≈ 47%
```

Rounded to **~50%** to reflect that several "Substantially Addressed"
items (R3-Issue1/3/5, R3-Minor9, R2-MC1/MC2) are close to done modulo a
disclosed, deliberate cap256 scope decision rather than a real gap. The
binding drag on this number is the **4 NOT ADDRESSED items** (R3-Issue6,
R3-Minor8, R3-Rec5, R3-Rec6 — fallback over-emphasis and shortening), which
all have completed decisions/plans but zero applied edits.

## 2. Manuscript readiness: **~65%**

`main.tex` compiles cleanly (42 pages, zero undefined refs, confirmed in
an earlier pass) and contains real, substantive content for nearly every
reviewer concern — not placeholders. What keeps this below ~80%:

- Fallback still occupies a numbered Contributions slot despite a decision
  to demote it (Task 2) — not yet edited.
- HALP differentiation is thin and Limitations contains no disclosure of
  the comparison's scope limit (Task 3) — not yet edited.
- Zero shortening cuts applied; word count grew 31% in this revision pass
  against an explicit 30-40% *reduction* demand (Task 1) — not yet edited.
- A dedicated overclaiming/hedging re-read (flagged as outstanding by the
  response skeleton itself for R3-Issue4, and by this audit for R3-Rec4's
  hedging-reduction half) has not been performed.
- The AI Declaration section has not been cross-checked against the
  validation-methodology argument the response letter relies on for
  R3-Issue7 (confirmed by direct read in this pass: lines 640-642 are a
  generic 3-sentence tool-disclosure, not the richer argument).

None of these require new data or compute — all are writing/editing tasks
against already-completed analysis, which is why the number is closer to
65% than to 30-40%.

## 3. Response-letter readiness: **~35%**

Per the response skeleton's own status summary: 0 `[DONE]`, 13
`[IN PROGRESS]`, 4 `[PENDING BASELINE DECISION]`, 3
`[PENDING MANUSCRIPT REWRITE]`. Substantive draft prose exists for every
row, but:

- The HALP paragraph (R3-Issue2) has a literal unfilled placeholder
  bracket, despite ready replacement text existing since 2026-06-19
  (Task 3 finding).
- The fallback paragraph (R3-Issue6/R3-Rec5) states the demotion as done
  but does not yet reflect this pass's specific decision (KEEP AS OPTIONAL
  IMPLEMENTATION DETAIL, with concrete removal-from-Contributions framing).
- No AE-level cover-letter-style roll-up paragraph has been drafted yet (it
  explicitly depends on the items below it, which are not yet finalized).
- R3-Summary's synthesis paragraph is an unfilled bracket pending the items
  above.
- No section has been independently proofread/finalized as submission text
  — every "IN PROGRESS" tag indicates real but provisional content.

## 4. Submission-package readiness: **~20%**

Directly re-checked in this pass:

- `submission_kbs_revision_docx/` exists with 6 draft skeleton `.docx`
  files (cover letter, highlights, CRediT, declaration of interest, author
  agreement, response-to-reviewers), all generated 2026-06-19 via pandoc.
- **The `response_to_reviewers_skeleton.docx` in that directory is now
  stale**: its markdown source differs from the current, actively-updated
  `reports/kbs_response_to_reviewers_skeleton.md` (confirmed via `diff` in
  this pass — files do not match), and the `.docx` was never regenerated
  after the 2026-06-19 23:39 build, so it predates the entire cap128 update
  and this audit pass's three new decisions.
- The cover letter skeleton is a draft for a **revision** cover letter;
  the only cover letter that exists outside this skeleton directory
  (`cover-letter.tex` in the repo-root submission zip) is written for an
  **initial** submission and would need full rewriting, not reuse.
- **No revised-manuscript DOCX or submission-ready PDF exists anywhere** —
  confirmed absent in the 2026-06-19 inspection and not contradicted by
  anything found in this pass.
- No Highlights file, CRediT statement, or standalone Declaration of
  Interest exists outside the draft-skeleton directory; none have been
  promoted to final form.

## 5. Overall readiness: **~48%**

Weighted average (manuscript and reviewer-concerns weighted highest since
they are substantive; response-letter and submission-package weighted
lower since they are comparatively mechanical/derivative once the
manuscript and decisions stabilize):

```
0.35 × manuscript(65%) + 0.30 × reviewer-concerns(50%)
  + 0.25 × response-letter(35%) + 0.10 × package(20%)
= 22.75 + 15.0 + 8.75 + 2.0 = 48.5%  →  ~48%
```

This should be read as "roughly half the distance from the post-cap128
manuscript state to a submittable revision," not as a literal probability
of acceptance — the remaining 52% is concentrated in well-specified,
already-analyzed editing tasks (no remaining unknowns requiring new
experiments), which is a favorable risk profile for the 2026-07-08
deadline.

## 6. Top 10 remaining tasks, ranked by importance

1. **Apply the fallback decision to `main.tex`** (remove from numbered
   Contributions list; shrink Robust Decision Mechanism subsection; fix
   Table 6's "EV" row text) — closes R3-Issue6/Rec5 and the fallback half
   of R2-MC3, the most direct, unconditionally-worded reviewer demand left
   unresolved. Decision and exact actions: `kbs_fallback_final_decision_report.md` §5.
2. **Apply the HALP positioning edits** — replace the thin Related Work
   sentence with the sharper pairwise/pointwise + factual/counterfactual
   differentiation, and add the Limitations disclosure sentence. Exact
   text ready to paste: `kbs_halp_positioning_final_audit.md` §7.1-7.2.
3. **Fill the response-letter's HALP placeholder bracket** (R3-Issue2,
   lines 250-265 of the skeleton) with the already-drafted honest-disclosure
   text — `kbs_halp_positioning_final_audit.md` §7.3. Five-minute edit,
   currently blocking that row's status from advancing.
4. **Execute the manuscript shortening cuts** (R3-Minor8/Rec6) — apply
   Cuts 1-9 from `kbs_manuscript_shortening_execution_plan.md`; expect
   ~19-25% reduction, short of 30-40%, and disclose that gap honestly in
   the response letter rather than overstate the result (plan §6).
5. **Perform a dedicated overclaiming/hedging re-read** of the full
   manuscript — distinct from the shortening pass; targets residual
   overclaiming language (R3-Issue4's self-flagged open item) and measures
   whether hedging density actually fell (R3-Rec4's still-open half).
6. **Cross-check or extend the AI Declaration section** in `main.tex`
   against the validation-methodology argument the response letter makes
   for R3-Issue7 (schema verification script, cap128 anomaly audit) —
   currently a generic 3-sentence tool list with no link to that argument.
7. **Draft the AE roll-up response paragraph and R3-Summary synthesis
   paragraph** — both are explicitly gated on items 1-6 above; do last
   among the writing tasks, not first.
8. **Write an actual revision-specific cover letter** — none exists; the
   only one in the repo (`cover-letter.tex`) is for the original
   submission and needs a full rewrite, not light editing.
9. **Regenerate the submission-package `.docx` skeletons** in
   `submission_kbs_revision_docx/` from the current, post-edit
   `main.tex`/response-letter content — the existing ones are confirmed
   stale (pre-dating cap128 and all three decisions made in this pass).
10. **Recompile `main.tex` with `tectonic`** after all edits above and
    re-verify a clean compile (no undefined refs, sane page count) before
    treating the manuscript as submission-ready — a cheap, mechanical
    final check that should not be skipped after a multi-edit pass.

None of these 10 tasks require new compute, a new baseline, or new
experimental evidence — every one is a writing, editing, or
packaging action against analysis that already exists in this repo, which
is consistent with this turn's "no new heavy experiments" constraint.
