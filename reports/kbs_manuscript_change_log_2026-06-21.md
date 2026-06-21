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
