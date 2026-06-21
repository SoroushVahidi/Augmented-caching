# Manuscript change log — revision-writing pass, 2026-06-21

Scope: convert the completed cap32/cap64/cap128 end-to-end evidence and the
cap128 anomaly sanity audit into actual `manuscript_source/main.tex` text.
cap256 was **not** launched and is **not** claimed anywhere below. No raw
CSV/MD output was modified; no push/commit/merge/delete/rename was
performed. This log lists every edit applied to `main.tex` in this pass, in
document order, with the rationale for each.

## 1. Abstract

Appended two sentences after the existing offline-ablation summary:
- States that an end-to-end online replay evaluation was added across three
  cache capacities (32, 64, 128 slots) over all seven trace families.
- States plainly that `evict_value_v1` does **not** outperform LRU, SIEVE,
  or FIFO-Reinsertion in this evaluation, that the gap widens
  non-monotonically at the largest evaluated capacity, and that this is
  linked to the single-step continuation rule used to build the
  supervision target.
- Reframes the overall claim: finite-horizon eviction-value prediction is a
  well-motivated, learnable supervision-target design; the present
  instantiation of the policy is not yet a practically superior online
  replacement rule relative to strong lightweight baselines.

Rationale: directly answers R3-Issue4/R3-Rec4 (self-admitted weakness vs.
contribution framing) at the point reviewers read first.

## 2. New subsection: "End-to-End Online Replay Evaluation Across Available Capacities"

Inserted before "Discussion and Analysis," with label
`subsec:end_to_end_capacities`. Contents:
- Three paragraphs explaining the evaluation setup (cap32/64/128, all seven
  families, the canonical 8-policy roster, explicit statement that cap256
  was not run and is out of scope for this revision).
- A table (`tab:available-capacities-trend`) of mean replay misses for all
  8 policies at all 3 capacities, values taken directly from
  `analysis/kbs_policy_trend_available_capacities.csv` /
  `reports/manuscript_artifacts/kbs_policy_trend_available_capacities.md`
  (LRU 34667.1/33650.1/32495.6; SIEVE 35313.6/34362.7/33279.3;
  FIFO-Reinsertion 34688.0/33672.1/32530.6; PredMk 34843.9/33773.1/32664.6;
  T&D 35087.0/33998.7/32928.9; BO/LRU 34668.6/33651.3/32496.9; REST
  34667.1/33650.1/32495.6; EV 36344.1/34409.0/36316.6).
- A figure (`fig:available-capacities-trend`,
  `figures/figure_available_capacities_trend.png`) with two panels:
  (a) mean misses per policy by capacity, (b) `evict_value_v1`'s gap vs
  LRU/SIEVE/FIFO-Reinsertion by capacity.
- Discussion paragraphs stating the exact gaps (vs LRU: +4.84%/+2.26%/
  +11.76% at cap32/64/128; vs SIEVE: +2.92%/+0.13%/+9.13%; vs
  FIFO-Reinsertion: +4.77%/+2.19%/+11.64%) and noting the other baselines
  cluster tightly around LRU while `evict_value_v1` does not.

Rationale: directly answers R2-MC3, R3-Issue1, R3-Issue3, R3-Rec1, R3-Rec2
(end-to-end results were entirely absent from the original submission).

## 3. New subsection: "Workload-Specific Breakdown"

Inserted immediately after the subsection above, label
`subsec:workload_breakdown`. Contents:
- A table (`tab:family-capacity-gap`) of `evict_value_v1`'s per-family gap
  vs LRU at cap32/64/128 for all 7 families, computed directly from the
  three chunk CSVs (BrightKite +10.47/+10.72/+50.91; Citi Bike
  +7.90/+4.70/+45.49; CloudPhysics +0.72/+1.57/+3.10; MetaCDN
  +22.37/+2.73/+17.62; MetaKV +1.70/+0.01/+0.76; Twemcache
  +1.84/+3.16/+14.06; Wikimedia +0.00/+0.00/+0.00, with a footnote
  explaining the Wikimedia trace is degenerate — LRU already achieves a
  100% hit ratio at all three capacities, so no policy can be
  differentiated on it).
- Discussion of the wiki2018 degeneracy, the comparative stability of
  CloudPhysics/MetaKV, and the distinct shapes of the MetaCDN (U-shaped:
  worst at cap32, best at cap64, bad again at cap128) and Twemcache
  (steadily increasing) degradation patterns.
- An explicit statement that the capacity-128 degradation is broad-based
  (4 of 6 non-degenerate families show double-digit or near-double-digit
  gaps), not an artifact of two outlier traces — a more accurate
  characterization than the original two-family framing in
  `reports/kbs_cap128_anomaly_sanity_audit.md`, reached by computing exact
  per-family numbers directly from the chunk CSVs rather than relying on
  the audit's headline framing alone.

Rationale: directly answers R3-Minor9/R3-Rec8 (missing workload-specific
analysis).

## 4. Discussion and Analysis

Replaced two paragraphs:
- Added an explicit statement that the end-to-end evidence above directly
  answers, in the negative, whether `evict_value_v1` is competitive online.
- Added a new paragraph proposing the causal mechanism for the cap128
  anomaly: the offline supervision target is built under a single-step
  LRU-continuation counterfactual, but the deployed policy recursively
  makes its own non-LRU eviction choices, so the realized trajectory
  diverges from the label-construction assumption — plausibly more severely
  at larger capacities because more candidates are scored per decision.
- Revised the closing paragraph to characterize `evict_value_v1`, in its
  present form, as a decision-aligned supervision study rather than a
  practically superior online policy, while stating that SIEVE and
  FIFO-Reinsertion remain strong, low-overhead baselines the method does
  not currently outperform.

Rationale: directly answers R3-Issue4 and gives the cap128 anomaly an
honest, evidence-grounded explanation rather than leaving it unexplained.

## 5. Limitations, point 3

Replaced the old hedge ("the empirical evidence currently shown in the
manuscript is stronger for scorer construction... depend on stronger
end-to-end evaluation evidence than is presently displayed" — written before
any end-to-end evaluation existed) with a substantially longer paragraph
that:
- States the actual finding: no improvement vs. LRU/SIEVE/FIFO-Reinsertion
  at any of cap32/64/128, with a sharp non-monotonic widening at cap128
  that is broad-based across 4 of 6 non-degenerate families.
- Lists four supporting diagnostic checks from the anomaly audit (clean raw
  outputs; deterministic small-scale reproduction; no capacity-specific
  code branching; training-set family imbalance that is constant, not
  capacity-growing, for the two most-affected families).
- States the leading (still unconfirmed) hypothesis: compounding
  distribution shift from the single-step LRU-continuation label.
- States what evidence would be needed to confirm the hypothesis, and that
  cap256 is on hold pending further analysis, not abandoned.
- Closes with the same decision-aligned-supervision-study framing used in
  the Discussion section, for consistency.

Rationale: directly answers R3-Issue4/R3-Rec4 in the section reviewers
expect this kind of finding to be confronted most directly.

## 6. Summary of Findings, paragraph 1

Softened "but a practically useful basis for learned cache replacement" to
"but a productive object of study for learned cache replacement," and added
a sentence distinguishing this offline finding from end-to-end online
competitiveness, which is evaluated separately and is not established by
the offline ablation alone.

Rationale: prevents the offline ablation from being read as already
implying online competitiveness, ahead of the negative end-to-end result
that follows.

## 7. Summary of Findings, closing paragraph

Replaced "Overall, the results support a focused conclusion... stronger
claims... should be reserved for future evaluation with more comprehensive
artifact-backed evidence" (written when no end-to-end evidence existed)
with "Overall, the results support a focused but mixed conclusion,"
explicitly restating the negative end-to-end finding and the
decision-aligned-supervision-study framing, consistent with the Abstract,
Discussion, and Limitations edits above.

Rationale: the manuscript must not contain a Conclusions paragraph that
implies a stronger result than what Results/Discussion now show.

## 8. Figure path fix

`\includegraphics` for the new figure initially relied on the repo-root
manuscript-artifact path, but `manuscript_source/figures/` is a separate,
flat directory containing the local copies used by `main.tex`. The
manuscript-facing figure is therefore copied into
`manuscript_source/figures/figure_available_capacities_trend.png`, and
`main.tex` now references `figures/figure_available_capacities_trend.png`
to match the convention used by the other figures in the document. The
older `*_DRAFT` figure/snippet files are preserved as historical/internal
artifacts, but the final-facing manuscript path no longer depends on them.

## What this pass did NOT do

- Did not launch cap256, run any heavy experiment, or imply a cap256 result
  anywhere in the text.
- Did not touch `tables/manuscript/table3_main_quantitative_comparison.csv/.tex`
  (the separate, intentionally-stubbed canonical Table 3 pathway gated on a
  single capacity-blind merged CSV that does not exist) — left in its
  honest `NOT_VERIFIED` state.
- Did not overwrite any raw chunk CSV/MD (`*_cap32/64/128_with_sieve_fifo.*`).
- Did not apply the still-pending manuscript-shortening pass (R3-Minor8/
  R3-Rec6) — the manuscript grew in this pass, it was not shortened.
- Did not resolve the fallback validate-or-remove decision (R3-Issue6/
  R3-Rec5) or the HALP-reimplementation scope decision (R3-Issue2/R3-Rec2)
  — both remain open author decisions.

## Post-edit verification (same pass)

Re-ran `tectonic main.tex` from a clean state (removed `main.pdf/.aux/.log/
.bbl/.blg/.out` first). Result: exit 0, `main.pdf` written (42 pages,
890,312 bytes), zero `undefined` reference/citation warnings in the final
pass, only pre-existing cosmetic overfull/underfull `\hbox` warnings (not
introduced by this pass). Independently confirmed all five new labels
(`subsec:end_to_end_capacities`, `subsec:workload_breakdown`,
`tab:available-capacities-trend`, `fig:available-capacities-trend`,
`tab:family-capacity-gap`) are each defined exactly once via
`grep -n "\\label{...}" main.tex`. Also re-ran
`scripts/paper/verify_kbs_policy_chunks.py` on the cap32+cap64+cap128
chunks together: **PASSED**, zero errors/warnings.

## Claim-audit follow-up (conservative wording pass)

A later text-only follow-up pass applied the manuscript/response claim-audit
recommendations without changing any raw evaluation artifacts:

- Removed the guarded fallback mechanism from the main numbered
  contributions list in `manuscript_source/main.tex`.
- Simplified the EV roster entry to "Candidate-level finite-horizon
  eviction-value predictor" rather than bundling the optional guard into
  the core method description.
- Softened implications-language that previously gave the guard more
  conceptual weight than the evidence supports; the guard is now described
  only as an optional unvalidated extension, not as support for the
  quantitative claims in this revision.
- Made the three-capacity replay framing more explicit in the manuscript
  and response drafts, including that `evict_value_v1` does not outperform
  LRU, SIEVE, FIFO-Reinsertion, or REST on the available-capacity average,
  and that cap256 remains unevaluated and unclaimed.
- Replaced the final-facing available-capacity figure/snippet filenames
  with non-`_DRAFT` names (`figure_available_capacities_trend.*`) while
  preserving the older `*_DRAFT` copies as historical/internal artifacts.
- Removed stale "final sweep" / "canonical result" placeholders from the
  response-to-reviewers drafts and replaced hard-coded table/figure numbers
  with scope-stable subsection descriptions where appropriate.

This follow-up did **not** regenerate figures/tables, run experiments,
launch cap256, create the absent canonical heavy-r1 CSV, or change any raw
CSV/MD evidence files.

## Title and AI Declaration follow-up (no-compute pass)

A further text-only pass, made after a reviewer-satisfaction audit of PR #49,
applied two remaining no-compute fixes:

- Changed the title from "Decision-aligned eviction-value prediction for
  robust learning-augmented caching" to "Decision-aligned eviction-value
  prediction for learning-augmented caching," since the guard/fallback
  robustness mechanism is explicitly unvalidated in this revision and the
  prior title echoed R3-Issue4's original complaint that the title promises
  more than the evidence supports.
- Expanded the AI Declaration to state that AI tools were also used to
  audit repository artifacts and check manuscript/code/result consistency,
  that all AI-assisted text, code, and analysis were independently verified
  by the author against the repository's artifacts (including the
  schema-validation, replay-artifact, and negative-result audits already
  documented elsewhere in the repository), and that AI assistance was not
  used to introduce any positive result unsupported by those artifacts.

This pass did not change any raw evaluation artifact, did not launch
cap256, and did not alter the reported three-capacity end-to-end result.

## Residual robustness-framing follow-up (no-compute pass)

A further text-only pass removed the two remaining over-strong uses of
"robust" identified by the prior audit, both outside the title:

- Changed the `\keyword{}` field entry "robust cache control" to "caching
  baselines," since the guard/fallback mechanism that the original phrase
  implicitly referenced is unvalidated in this revision.
- Changed the Introduction sentence "We focus on robust learning-augmented
  caching in the unweighted paging setting..." to "We focus on
  learning-augmented caching under strong heuristic and combiner
  baselines in the unweighted paging setting...," for the same reason.

Other uses of "robust"/"robustness" elsewhere in the manuscript were left
unchanged because they either describe the literature category of other
baselines (e.g., "robust reference policies," "robust combiner-style
baselines" for REST/T&D/BO-LRU) or are already-hedged limitation language
about the guard (e.g., "not a theorem-backed robustness guarantee," "remains
heuristic"); none of these claim a demonstrated robustness result for
`evict_value_v1`.

This pass also reviewed the existing `\section*{Acknowledgements}` block
(already present, already thanking Professor Ioannis Koutis and the Wulver
HPC system) but made no change to it; any further additions are left as an
author decision rather than an automatic edit.

This pass did not change any raw evaluation artifact, did not launch
cap256, and did not alter the reported three-capacity end-to-end result.

## Guard subsection rename (no-compute pass)

A final-audit pass identified one remaining structural echo of the
title's robustness overclaim: the Method subsection describing the
optional fallback mechanism was headed "Robust Decision Mechanism"
(`\label{subsec:robust_decision_mechanism}`), while its own body text
states the guard is "not a theorem-backed robustness guarantee" and that
its empirical effect "has not yet been measured." This text-only fix:

- Renamed the subsection to "Guarded Fallback Mechanism"
  (`\label{subsec:guarded_fallback_mechanism}`).
- Updated the one descriptive cross-reference immediately before the
  subsection ("...forms the basis for the robust decision mechanism
  described next" to "...the guarded fallback mechanism described
  next").

No `\ref{subsec:robust_decision_mechanism}` cross-reference existed
elsewhere in the document, so no other text required updating. No
results, numbers, tables, cap256 language, or baseline/fallback claims
were changed.

This pass did not change any raw evaluation artifact, did not launch
cap256, and did not alter the reported three-capacity end-to-end result.

## Overhead benchmark added (controlled timing pass)

A new controlled per-decision wall-clock latency benchmark
(`scripts/run_overhead_benchmark.py`,
`analysis/kbs_overhead_benchmark_local_tmux_20260621.csv/.md`) was run to
fill the gap the Overhead and Scalability subsection had explicitly
flagged as an open item ("we have not yet produced a controlled,
capacity-isolated wall-clock benchmark"). Key facts:

- Measured on the author's local/cloud development machine under tmux —
  not on Wulver and not under Slurm.
- Trace: a 5,000-request prefix of BrightKite (`brightkite_50k`), not the
  full seven-trace canonical sweep.
- Capacities: 32, 64, 128. No cap256.
- Policies: lru, sieve, fifo_reinsertion, rest_v1, evict_value_v1 — all
  five completed with zero failures.
- Result: `evict_value_v1` mean per-eviction-decision cost is 75.0\,ms /
  152.1\,ms / 316.0\,ms at capacities 32/64/128 (near-linear in capacity,
  consistent with the existing $O(k)$ argument), versus ~0.001\,ms for
  LRU/FIFO-Reinsertion, 0.002--0.005\,ms for SIEVE, and 0.04--0.18\,ms for
  REST.
- Does not touch, modify, or create
  `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` (that
  file does not exist anywhere in the repository).
- Does not claim `evict_value_v1` outperforms any baseline; this is a cost
  measurement only.

`main.tex`'s Overhead and Scalability subsection was updated to replace
the "open item" sentence with the measured numbers above (and the
now-resolved TODO comment above it was removed). The response-letter
skeleton's R2-MC2, Issue 5, and Rec 3 entries were each given a short
"Update 2026-06-21" note plus the same numbers; none of their status tags
were bumped to `[DONE]`, consistent with that file's own caution that
ready evidence still needs a final cross-check pass.
`submission_kbs_revision_docx/response_to_reviewers_skeleton.md` is a
separately-organized, condensed draft (not a structural mirror of the
`reports/` skeleton) and still contains the older "we do not claim a
controlled wall-clock timing benchmark" wording; it was intentionally
left unedited in this pass and is flagged here as a known follow-up
rather than silently left inconsistent.

This pass did not change any raw evaluation artifact other than the three
new overhead-benchmark files listed above, did not launch cap256, and did
not alter the reported three-capacity end-to-end result.

## Repetition/verbosity pass (no-compute, R3-Minor8/R3-Rec6)

A narrow, no-compute pass targeting R3-Minor8/R3-Rec6 (verbosity/repetition)
removed three near-duplicate restatements of already-established claims,
without changing any claim, number, or caveat:

- End of "Workload-Specific Breakdown": replaced a full restatement of the
  "decision-aligned supervision target... not yet a practically superior
  online policy... does not improve on LRU/SIEVE/FIFO-Reinsertion" sentence
  with a shorter bridge sentence plus explicit cross-references to
  Section~\ref{subsec:discussion_analysis} (Discussion) and
  Section~\ref{subsec:limitations} (Limitations), where the full claim is
  stated once each and left untouched.
- End of "Summary of Findings": condensed a paragraph that nearly duplicated
  the closing paragraph of "Discussion and Analysis" verbatim, replacing it
  with a shorter version that keeps the mixed-conclusion summary and the
  negative end-to-end result but cross-references Discussion instead of
  re-deriving it.
- "Implications of the Proposed Approach": merged two near-identical
  guard/fallback caveats ("not used to support the quantitative claims in
  this revision") that appeared in two separate paragraphs of the same
  subsection into one, with the second occurrence now cross-referencing the
  first and Section~\ref{subsec:guarded_fallback_mechanism}.

The Abstract, the full "Discussion and Analysis" closing paragraph, and the
full Limitations capacity-128 paragraph were left fully intact, as were all
numerical results, the cap256/canonical-heavy-r1 caveats, the guard/fallback
unvalidated framing, and the overhead-benchmark local/tmux (not
Wulver/Slurm) description. Net effect: 3 paragraphs edited, ~115 words
removed, no new experiments, no claim changes.

## Final co-author-readiness pass (no-compute)

A second narrow, no-compute pass closed three remaining items from a final
readiness audit of PR #49:

- Trimmed the overlap between "Problem Setting and Motivation" (1.1) and
  "Research Objective and Scope" (1.2) that R3-Minor8 named explicitly: the
  candidate-level framing and the "predictive information is useful only if
  it improves the decision" premise were each stated twice and are now
  stated once, with 1.1 closing with a cross-reference to 1.2 instead of
  re-deriving 1.2's content. Also replaced one residual "translated into
  robust online eviction decisions" phrase in 1.2 with "translated into more
  decision-aligned online eviction choices" to avoid echoing the
  title-level robustness framing already removed elsewhere.
- Added one Limitations sentence (new "Sixth" point) stating that this is a
  single-author revision whose validation cannot substitute for independent
  multi-author replication, mitigated by repository-level audit trails,
  schema/replay-artifact checks, and negative-result reporting. This gives
  the manuscript body the in-text counterpart that the R3-Issue7 response
  already referenced but that previously existed only in the back-matter AI
  Declaration.
- Updated the R3-Minor8/R3-Rec6 text in both
  `reports/kbs_response_to_reviewers_skeleton.md` and
  `submission_kbs_revision_docx/response_to_reviewers_skeleton.md`, which
  had still said a shortening pass was "planned" and "not yet applied to
  `main.tex`" — stale as of the previous pass's commit. Both now describe
  the two narrow passes actually applied and are explicit that the full
  30-40% reduction target is still open. Status tags for R3-Minor8/R3-Rec6
  moved from `[PENDING MANUSCRIPT REWRITE]` to `[IN PROGRESS]` in the
  tracker (R3-Summary stays `[PENDING MANUSCRIPT REWRITE]`, since it
  depends on the full R3-Rec6 target, not just the narrow passes); the
  status-summary table's `[IN PROGRESS]` count was also corrected from a
  pre-existing miscount (listed 16 items under "13") to the actual count.

No experiments were run, no raw evaluation artifact changed, and no claim,
number, or caveat (cap256, canonical heavy-r1, guard/fallback validation
status, overhead-benchmark environment, negative end-to-end result) changed.
Net effect: 2 short paragraphs trimmed/merged in the Introduction, one new
Limitations sentence added, and the two response-letter files' R3-Minor8
sections brought up to date with the manuscript's actual state.

## Submission-package finalization pass (no-compute)

This pass finalized the submission-facing response letter and built the
manuscript PDF, with PR #49's prior commits already incorporated into
`main` and pushed to `origin/main`:

- Removed a dead, never-rendered `% TODO(revision): ...` LaTeX comment block
  from the "Replay Horizon Selection" subsection of `main.tex`; the rendered
  sentence immediately following it already states the same open question
  honestly, so the comment was pure stale scaffolding with no effect on the
  PDF.
- Rewrote `submission_kbs_revision_docx/response_to_reviewers_skeleton.md`
  to remove internal tracker/process language so it reads as a final
  response letter rather than an internal draft: dropped the
  "DRAFT — NOT FINAL" / "source of truth tracker" header, the dated
  "2026-06-21 update:" changelog banner (its non-redundant content was
  already duplicated in Issue 1/Issue 3), and the "Do not submit as-is"
  gating section. Reworded "pending a separate, explicit scope decision,"
  "not yet resolved," "out of scope for this revision cycle," and "candidate
  follow-up before final submission" into direct, honestly-scoped statements
  (e.g., "outside the scope of this revision," "remains available as a
  further revision if the Editor and reviewers consider it necessary").
  Added an explicit cross-reference in the Issue 7 response to the
  manuscript's new Limitations sixth point. Replaced the removed gating
  section with a "Closing Note" that names the three items still explicitly
  out of scope (fallback validation ablation, faithful HALP reimplementation,
  full 30-40% shortening target) as disclosed limitations rather than
  blockers. No reviewer comment, claim, or number was added or removed in
  this rewrite; `reports/kbs_response_to_reviewers_skeleton.md` (the internal
  tracker) was intentionally left untouched and keeps its status-tag
  structure.
- Rebuilt `manuscript_source/main.pdf` via `tectonic main.tex` from a clean
  state: exit 0, 41 pages, 889,424 bytes, zero undefined-reference/citation
  warnings, only routine overfull/underfull `\hbox` warnings.
- Added `submission_kbs_revision_docx/KBS_revised_manuscript_for_visual_check_20260621.pdf`,
  a copy of the same PDF, for visual/external-AI review. No `.docx` file was
  created or modified.

No experiments were run, cap256 was not launched, and
`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` was not
created or touched (confirmed absent both before and after this pass).

## Figure polish pass (no-compute, Fig. 3 visual cleanup)

A visual-check of the rebuilt PDF on page 19 found two cosmetic issues in
Figure 3 (the available-capacity replay figure). Both were fixed at the
source (`scripts/paper/build_kbs_available_capacities_figure.py`) and the
figure was regenerated from the same, unchanged input CSV
(`analysis/kbs_policy_trend_available_capacities.csv`) — no underlying
numbers changed:

- Removed the red `fig.suptitle(...)` ("Available-capacity replay only
  (capacities 32, 64, 128; cap256 not evaluated)") that duplicated
  information already stated in the caption and looked like an unresolved
  draft marker. The caption already states the evaluated/not-evaluated
  capacities, so no replacement title was added.
- In panel (b), the `vs LRU` and `vs FIFO-Reinsertion` gap curves are
  visually close because the two baselines have nearly identical mean miss
  counts (within 0.11% of each other at every capacity, as already noted in
  the body text). Gave each of the three baselines (`vs LRU`, `vs SIEVE`,
  `vs FIFO-Reinsertion`) a distinct line style (solid/dotted/dashed) and
  marker (circle/triangle/square) plus a properly capitalized legend label,
  and added one sentence to the figure caption in `main.tex` explaining the
  near-coincidence so a reader does not mistake it for a plotting error.
- Regenerated `figures/manuscript/figure_available_capacities_trend.pdf`,
  `figures/manuscript/figure_available_capacities_trend.png`,
  `manuscript_source/figures/figure_available_capacities_trend.png`, and
  the (unused-by-`main.tex`) companion snippet
  `reports/manuscript_artifacts/latex_snippets/figure_available_capacities_trend_snippet.tex`
  via `PYTHONPATH=scripts/paper python3
  scripts/paper/build_kbs_available_capacities_figure.py`. Rebuilt
  `manuscript_source/main.pdf` via `tectonic main.tex`: exit 0, 41 pages,
  889,274 bytes, zero undefined-reference/citation warnings, only routine
  overfull/underfull `\hbox` warnings. Visually re-checked page 19 of the
  rebuilt PDF: the red label is gone and the three gap-curve styles/markers
  are distinguishable in the legend. Updated
  `submission_kbs_revision_docx/KBS_revised_manuscript_for_visual_check_20260621.pdf`
  to match.

This pass did not run any experiment, did not launch cap256, did not touch
or create `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`,
and did not change any reported number, gap percentage, or claim — only the
figure's visual styling and one caption sentence changed. The separate,
pre-existing `*_DRAFT` figure/snippet files
(`figures/manuscript/figure_available_capacities_trend_DRAFT.*`,
`manuscript_source/figures/figure_available_capacities_trend_DRAFT.png`,
`reports/manuscript_artifacts/latex_snippets/figure_available_capacities_trend_DRAFT_snippet.tex`)
are untracked, unreferenced by `main.tex`, and were left untouched.
