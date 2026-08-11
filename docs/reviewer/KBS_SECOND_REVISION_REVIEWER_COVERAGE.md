# KBS Second-Revision Reviewer Coverage Map

Status: authoritative local source of truth for reviewer-concern coverage.

This file consolidates the reviewer-concern tables previously spread across
`docs/reviewer/kbs_second_revision_artifact_map.md`,
`docs/reviewer_revision_roadmap.md`, and
`analysis/kbs_local_current_evidence_synthesis_20260810/reviewer_evidence_map.csv`.
Those files should point here for the coverage matrix itself rather than
repeating it.

No raw reviewer-letter file exists in this local repository. The concern
structure below is the one this branch's own docs already track (`Reviewer #2
Major 1-4`, `Reviewer #3` continuation-mismatch) -- derived from the actual
review by earlier work on this branch, not reconstructed from memory here.
Cross-reference shorthand (`MC1..MC3`, `R3-Issue1..6`) used in some prior
internal notes is mapped by topic where it does not have a literal doc label;
this is noted explicitly per row rather than asserted as verified.

Statuses used: `ANSWERED`, `ANSWERED_WITH_CAVEATS`, `PARTIAL`, `RUNNING`,
`MISSING`, `TEXT_ONLY`.

Last updated: 2026-08-10, while `50%` learning-curve fraction is 4/7 folds
complete. Does not count unsynced Wulver numerical evidence as local.

---

## Reviewer #2 Major 1: learned-baseline comparison

- Concern paraphrase: does the method actually beat strong existing learned
  and non-learned caching baselines under a fair protocol (LRU, SIEVE, FIFO,
  LRB, 3L-Cache, HALP, CACHEUS, offline Belady as oracle context)?
- Evidence complete locally: baseline rows themselves are
  `COMPLETE_VALIDATED` for controlled-window rows
  (`analysis/reviewer_fairness/policy_comparison_*.csv`), cross-checked by
  `analysis/kbs_comparison_fairness_audit.json` (overall_score `76`).
- Known Wulver evidence from existing docs only: none asserted; held-out
  `evict_value_v1` retraining is explicitly `BLOCKED_PENDING_SYNC` /
  `PARTIAL` per the artifact map, not a Wulver-only gap.
- Running/pending evidence: corrected primary `evict_value_v1` head-to-head
  replay against these baselines remains `PARTIAL` (`analysis/reviewer_fairness_cross_family_v1/`
  not yet a usable primary table); the old fair-window comparison
  (`analysis/reviewer_fairness/policy_comparison_evict_value_v1.csv`) is
  `CONTAMINATED_DO_NOT_USE` due to recorded train/test overlap.
- Remaining experiment: complete cross-family held-out `evict_value_v1`
  replay with the frozen registry and local model hashes (fairness audit's
  top required fix, cost `high`, reviewer_value `very_high`).
- Text-only fix: label `offline_belady` explicitly as oracle context, not a
  deployable baseline, everywhere it appears (already flagged as required in
  the fairness audit; verify consistently applied).
- Status: `ANSWERED_WITH_CAVEATS` for the baseline pool itself;
  `PARTIAL` for the `evict_value_v1` head-to-head claim specifically.
  Per-baseline fidelity caveats: HALP `LOW_TO_MEDIUM`, LRB/3L-Cache
  `MEDIUM`, CACHEUS `HIGH`.

## Reviewer #2 Major 2: supervision-objective ablation

- Concern paraphrase: is the chosen supervision objective (`eviction_loss`)
  actually a good choice compared to plausible alternatives?
- Evidence complete locally: `COMPLETE_VALIDATED` --
  `analysis/supervision_objective_ablation_v1/policy_comparison.csv` (84
  rows, 7 families x 4 objectives x 3 capacities, all `status=ok`), 28-model
  registry frozen (`MODEL_SELECTION_FROZEN=true`, `28/28` models present).
  Aggregate mean miss_ratio: `objective_pairwise` `0.673` <
  `objective_reuse_distance` `0.680` < `objective_next_arrival` `0.682` <
  `objective_eviction_loss` `0.716`; `eviction_loss` is worst or tied-worst
  in every one of 7 families.
- Known Wulver evidence from existing docs only: none asserted for this
  concern.
- Running/pending evidence: the same-target scalar-vs-pairwise
  learning-convergence diagnostic (a *different*, complementary question --
  representation, not target construction) is `25_PERCENT_COMPLETE_VALID`,
  `50%` `RUNNING`, `100%` not started.
- Remaining experiment: none required to answer the original Major 2 concern
  itself (already `COMPLETE_VALIDATED`); the learning-curve work is
  additional depth, not a gap in this concern.
- Text-only fix: keep the naming distinction explicit in manuscript prose --
  `objective_pairwise` (different target construction) is not the same
  condition as `eviction_loss_pairwise` (same target, pairwise
  representation only); conflating them is an identified `claim we must not
  make` in the negative-results notebook.
- Status: `ANSWERED_WITH_CAVEATS` (pairwise-model-selection-semantics caveat
  noted in the fairness audit, score `86`).

## Reviewer #2 Major 3 and Reviewer #3: distribution-shift / continuation-mismatch diagnosis

- Concern paraphrase: does the mismatch between LRU-continuation label
  construction and learned-policy deployment (sequential distribution shift)
  explain some or all of the performance gap?
- Evidence complete locally: `PARTIAL` --
  `analysis/distribution_shift_ablation_v1/` covers only `metacdn`, `24/42`
  primary rows, `4/7` families. Trajectory divergence is large (97-99.8% at
  3 capacities) but downstream misses *worsened* under the one directional
  test run (`DAGGER_ITER1`) despite a reduced measured state-shift index --
  divergence exists, but the one causal-adjacent test does not show a simple
  fix.
- Known Wulver evidence from existing docs only: seven-family continuation
  work is described in local docs as Wulver-owned / pending sync-back of the
  Wulver-only runner and Slurm files; not asserted as a numerical result here.
- Running/pending evidence: continuation-policy C1/C2 causal ablation source,
  tests, and frozen config exist and are sync-ready
  (`src/lafc/continuation_policy_ablation.py`,
  `configs/continuation_policy_causal_ablation_v1.json`), but only
  `TINY_SMOKE_ONLY` (`decision_count=3`) has actually been run locally -- no
  full result.
- Remaining experiment: full 7-family C1/C2 causal ablation at Wulver scale.
- Text-only fix: none required beyond what is already stated; the local docs
  already avoid claiming continuation mismatch is proven or that DAgger fixes
  the gap.
- Status: `PARTIAL` (distribution-shift evidence) / `MISSING` (full-scale
  causal C1/C2 result, `LOCAL_IMPLEMENTATION_READY` only).

## Reviewer #2 Major 4: practical significance (computational cost)

- Concern paraphrase: is fine-grained candidate-level learned eviction
  computationally practical, or does its overhead undermine the contribution?
- Evidence complete locally: `SMOKE_ONLY` --
  `analysis/practical_significance_ablation_v1/exact_optimization_equivalence.json`
  shows `all_variants_exact_across_all_trace_capacity_pairs=true` with
  smoke speedups roughly `14.29x-99.99x`, but the artifact itself records
  `speedup_numbers_are_final_reviewer_evidence=false`.
- Known Wulver evidence from existing docs only: none asserted.
- Running/pending evidence: controlled final timing campaign is
  `PENDING_CONTROLLED_RUN`, no final result present locally.
- Remaining experiment: controlled timing campaign, kept separate from smoke
  conclusions.
- Text-only fix: continue to state smoke-only status explicitly wherever
  these numbers are cited.
- Status: `PARTIAL` (implementation-equivalence result exists and is
  usable as supporting evidence; final timing claim is `MISSING`).

## R3-Issue2 / R3-Issue3 (subset of Major 1): HALP and SIEVE/FIFO differentiation

- Concern paraphrase: is the comparison against HALP (a close learned-preference
  competitor) and simple non-learned baselines (SIEVE, FIFO) fair and
  well-differentiated?
- Evidence complete locally: `ANSWERED` for SIEVE/FIFO (`COMPLETE_VALIDATED`,
  controlled-window rows usable directly); `ANSWERED_WITH_CAVEATS` for HALP
  (`COMPLETE_VALIDATED` but fidelity flagged `LOW_TO_MEDIUM` -- no public
  official HALP code, frozen offline split adapts a continuous production
  algorithm).
- Known Wulver evidence: none asserted.
- Running/pending evidence: none beyond the Major-1 head-to-head gap above.
- Remaining experiment: none specific beyond Major 1's required fix.
- Text-only fix: keep the HALP fidelity caveat attached wherever HALP numbers
  are cited.
- Status: `ANSWERED` (SIEVE/FIFO); `ANSWERED_WITH_CAVEATS` (HALP).

## R3-Issue4 (cross-cutting): empirical weakness / contribution framing

- Concern paraphrase: given the negative/mixed empirical results, what is the
  paper's actual contribution and how should the weakness be framed honestly?
- Evidence complete locally: the interpretive backbone for this is
  `docs/reviewer/kbs_negative_results_interpretation.md` plus this file and
  `KBS_SECOND_REVISION_HYPOTHESIS_MAP.md` and the
  `kbs_local_current_evidence_synthesis_20260810/` artifact -- an internal
  notebook, not manuscript prose.
- Known Wulver evidence: not asserted.
- Running/pending evidence: depends on essentially every hypothesis in the
  hypothesis map; several (H5/H6/H8/H10/H11) have no full result yet.
- Remaining experiment: see hypothesis map's ranked next-experiment list.
- Text-only fix: this is fundamentally a text/interpretation task, not a
  missing-experiment task, once the hypothesis map is stable; draft
  manuscript claims must draw only from `9.10 Potential manuscript claims` /
  `9.11 Claims we must not make` in the negative-results notebook.
- Status: `TEXT_ONLY` (framing work), gated on `PARTIAL` underlying evidence.

## R3-Issue6: fallback mechanism

- Concern paraphrase: if the learned policy is unreliable in some states,
  should it fall back to a safe heuristic (e.g. LRU) rather than degrade
  performance?
- Evidence complete locally: none -- no fallback mechanism is implemented.
  Confidence-gated LRU fallback and margin-gated softmax are listed only as
  future diagnostic candidates in the negative-results notebook's
  "decision-rule branch" (tied to H7 in the hypothesis map).
- Known Wulver evidence: not asserted.
- Running/pending evidence: none.
- Remaining experiment: design and implement a fallback mechanism; this is
  new work, not a rerun of existing code.
- Text-only fix: none available -- this concern currently has no evidence to
  cite, local or Wulver.
- Status: `MISSING`.

---

## Summary

- Most complete concern: **Reviewer #2 Major 2 (supervision-objective
  ablation)** -- `COMPLETE_VALIDATED`, frozen 28-model registry, consistent
  7-family result.
- Most under-addressed concern: **R3-Issue6 (fallback mechanism)** -- no
  local evidence exists at all, not even a smoke-scale implementation.
- Second most under-addressed: **Reviewer #2 Major 3 / R3 continuation
  mismatch** -- real implementation exists but only smoke-tested
  (`decision_count=3`); the one directional test available (DAgger) muddies
  rather than confirms a simple story.
