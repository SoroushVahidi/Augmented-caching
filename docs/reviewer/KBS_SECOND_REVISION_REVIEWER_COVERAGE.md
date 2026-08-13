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

Statuses used: `ANSWERED`, `ANSWERED_WITH_CAVEATS`,
`EXPERIMENTALLY_COMPLETE_SYNTHESIS_PENDING`, `PARTIAL`, `RUNNING`,
`LOCAL_COMPLETE`, `WULVER_PENDING`, `MISSING`, `TEXT_ONLY`.

Last updated: 2026-08-13 (C0/C1/C2 and distribution-shift closure -- see the
Reviewer #2 Major 3 / Reviewer #3 section below). Prior update 2026-08-11
incorporated fresh Wulver-side facts
relayed by the user from a separately audited Wulver session -- these are
labeled `WULVER_ONLY_VALIDATED` per row below and are **not** independently
verified by this workstation (Wulver was not contacted directly). See
[`../CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md`](../CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md)
for full detail behind every Wulver-sourced claim in this file, and
[`../WULVER_TO_GITHUB_PROMOTION_QUEUE.md`](../WULVER_TO_GITHUB_PROMOTION_QUEUE.md)
for sync priority. Local learning-curve fraction `50%` is now complete and
validated: 7/7 families, 42/42 rows, all `status=ok`; stopping decision
`STOP_SAMPLE_SIZE_HYPOTHESIS`; `100%` intentionally not run.

---

## Reviewer #2 Major 1: learned-baseline comparison

- Concern paraphrase: does the method actually beat strong existing learned
  and non-learned caching baselines under a fair protocol (LRU, SIEVE, FIFO,
  LRB, 3L-Cache, HALP, CACHEUS, offline Belady as oracle context)?
- Evidence complete locally: baseline rows themselves are validated for
  controlled-window rows, cross-checked by
  `analysis/kbs_comparison_fairness_audit.json` (overall_score `76`) and
  the 2026-08-11 compact evidence package
  `analysis/kbs_r2_major1_evidence_prep_20260811/`.
- Known Wulver evidence (updated 2026-08-11, `WULVER_ONLY_VALIDATED`,
  relayed by the user, not independently verified here):
  - **The corrected held-out `evict_value_v1` head-to-head replay is
    COMPLETE**: 42/42 rows, 7 families x 3 capacities x 2 variants, all
    `ok`, SHA-256 `982bfdffdbd816b56c2eef86ecb730a1eb136b3f85e36ad533739e586fa0a296`,
    at `analysis/reviewer_fairness_cross_family_v1/evict_value_v1_final_42_20260810/policy_comparison.csv`
    (Wulver path, not yet synced locally). This resolves what was, until
    this update, the largest single gap in this concern.
  - Fresh local audit found complete exact controlled-window CSVs for the
    three modern learned baselines: LRB is
    `LOCAL_EXACT_PROTOCOL_VALIDATED`, 3L-Cache is
    `LOCAL_EXACT_PROTOCOL_VALIDATED_WITH_CAVEAT`, and CACHEUS is
    `LOCAL_EXACT_PROTOCOL_VALIDATED_WITH_PROVENANCE_CAVEAT`. Each has 42
    local rows, including 21 primary `[10000,50000)` controlled-window rows
    across all seven families and capacities `32/64/128`, all `ok`, with no
    duplicate keys, no NaN/Inf, and matching SHA-256 values. Wulver jobs
    `1171965`, `1171966`, and `1171967` remain `PENDING` because of
    maintenance, but they are now replication/config-audit follow-up rather
    than a blocker to local baseline availability unless their missing
    config JSON proves materially different.
  - LRU/SIEVE/FIFO and HALP-causal do not need the exact-protocol re-run
    (no training/split dependency for the first three; HALP's existing
    result is already counted as valid comparison evidence) -- their
    existing local `FINAL_VALIDATED` rows stand.
- Remaining task (updated 2026-08-13, later pass): none for evidence
  collection. The corrected 42/42 `evict_value_v1` result has been synced
  from Wulver, independently re-audited locally, and compared under an
  exact evaluation protocol against all seven baselines (see
  `reports/kbs_final_evidence_20260813/major1_reviewer_summary.md`).
  Remaining task is manuscript/rebuttal wording integration only. Sync jobs
  `1171965`-`1171967` later only for replication/config audit if their
  config later proves materially different.
- Text-only fix: label `offline_belady` explicitly as oracle context, not a
  deployable baseline, everywhere it appears (already flagged as required in
  the fairness audit; verify consistently applied).
- **Status (updated 2026-08-13, later pass): `SCIENTIFICALLY_COMPLETE_SYNTHESIS_READY`.**
  Scientific baseline work is complete locally; the corrected Wulver
  `evict_value_v1` 42/42 result has been synced (16/16 transfer hashes PASS)
  and independently re-audited locally (all structural/leakage/model-hash
  gates PASS). **A same-protocol comparison now exists and is validated**
  against all seven baselines (LRB, 3L-Cache, CACHEUS, HALP, LRU, SIEVE,
  FIFO-Reinsertion): each has 21/21 `primary_controlled_window` cells
  matching the treatment exactly by trace SHA-256, capacity, window, and
  metric, with `future_information=none` on every row. Under this
  comparison, `evict_value_v1` loses on a clear majority of matched cells
  (13-17 of 21) against every baseline -- see
  `reports/kbs_final_evidence_20260813/major1_reviewer_summary.md` and
  `major1_protocol_comparability.md`. An earlier version of this status line
  claimed no same-protocol comparison exists and that LRB/3L-Cache/HALP/
  CACHEUS have zero results under any protocol; that was **incorrect** --
  those claims came from `baseline_eligibility.csv`, a Wulver-produced
  artifact whose "zero results... under any protocol" wording is accurate
  only about Wulver's own filesystem, not this workstation's separately
  produced `analysis/reviewer_fairness/` results. The exact controlled-window
  LRB/3L-Cache/CACHEUS rows are locally validated; their Wulver jobs remain
  pending only as replication/config-audit follow-up. Per-baseline fidelity
  caveats unchanged: HALP `LOW_TO_MEDIUM`, LRB/3L-Cache `MEDIUM`, CACHEUS
  `HIGH` with the current live-source provenance caveat.

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
- Known Wulver evidence (updated 2026-08-11): **none belongs to this
  concern.** The Wulver horizon-sensitivity sweep (job `1169299`) and the
  broad target-degeneracy campaign (job `1169513`) are explicitly *not*
  part of Major 2 -- they answer "is `H=4` an adequate horizon for the
  chosen objective" (MC1 / H3-H4/H9-H11 in the hypothesis map), a different
  question from Major 2's "is `eviction_loss` the right objective compared
  to alternatives at all." Do not cite horizon-sensitivity progress as
  evidence toward closing or reopening this concern; see the correction
  note in `../DEVELOPMENT_STATUS.md` and this file's mapping above.
- Mechanistic follow-up evidence: the same-target scalar-vs-pairwise
  learning-convergence diagnostic (a *different*, complementary question --
  representation, not target construction) is `FINAL_50PCT_VALIDATED`.
  It disfavors the sample-size explanation within the tested `1%-50%`
  range and records `STOP_SAMPLE_SIZE_HYPOTHESIS`; `100%` is intentionally
  not run.
- Remaining experiment: none required to answer the original Major 2 concern
  itself (already `COMPLETE_VALIDATED`); the learning-curve work is
  additional depth, not a gap in this concern. Remaining work for this
  concern is manuscript integration, not further experimentation.
- Text-only fix: keep the naming distinction explicit in manuscript prose --
  `objective_pairwise` (different target construction) is not the same
  condition as `eviction_loss_pairwise` (same target, pairwise
  representation only); conflating them is an identified `claim we must not
  make` in the negative-results notebook.
- **Status (reconfirmed 2026-08-13, do not downgrade to `PARTIAL` on
  account of horizon-sensitivity work being unfinished -- that is a
  different concern): `SCIENTIFICALLY_COMPLETE_SYNTHESIS_PENDING`** (`COMPLETE_VALIDATED`
  / `FINAL_VALIDATED` for the intended objective-comparison scope itself;
  remaining work is manuscript integration only), `ANSWERED_WITH_CAVEATS` at
  the reviewer-coverage level (pairwise-model-selection-semantics caveat
  noted in the fairness audit, score `86`).

## Reviewer #2 Major 3 and Reviewer #3: distribution-shift / continuation-mismatch diagnosis

**Snapshot 2026-08-13 (supersedes the 2026-08-12 snapshot below): both
remaining local campaigns are now closed.** C0/C1/C2 continuation-policy
causal ablation and distribution-shift completion each passed a formal,
read-only post-completion integrity audit on 2026-08-13 and are
`FINAL_VALIDATED` (21/21 units / 7/7 folds, all integrity gates PASS; the
distribution-shift log-ordering anomaly noted in the prior snapshot was
diagnosed as `BENIGN_STDOUT_BUFFERING`, not a defect). All planned local
mechanistic science for this concern is now complete. Compact evidence:
`reports/kbs_final_evidence_20260813/` (`c0_integrity_summary.md`,
`distribution_integrity_summary.md`, `mechanistic_hypothesis_summary.md`,
`reviewer_mapping.md`); canonical raw sources remain
`analysis/continuation_policy_causal_ablation_production_v1/` and
`analysis/distribution_shift_ablation_v1/`.

**Final result:**
- C0/C1/C2 (H5, continuation-policy mismatch): `PARTIALLY_SUPPORTED`. C2
  (frozen-`pi1` continuation) improves over C1 (LRU continuation) in 13/21
  cells, ties in 3/21 (Wiki2018, degenerate 100%-miss cells), worsens in
  5/21; macro mean C2−C1 miss-ratio delta ≈ −0.0102; aggregate misses
  C0=565126, C1=601569, C2=592970. Strongest improvement: `metacdn`.
  Strongest counter-example: `brightkite` cap32 (+0.2433, the single
  largest effect in the table, opposite direction) -- this blocks a
  uniform/universal claim; the result is regime-dependent, not
  universally causal.
- Distribution-shift (H6, generic state-shift): `DISFAVORED` as a
  shift-reduction-improves-performance story. DAgger improves misses in
  only 2/21 cells, ties in 3/21 (Wiki2018), worsens in 16/21; macro mean
  DAgger−OFF delta ≈ +0.0094 (net worse); aggregate misses OFF=591604,
  DAGGER=599537. The state-shift index itself *does* improve (decreases)
  in 16/21 cells -- the shift-reduction mechanism works as designed, but
  in 13/18 informative cells shift improves while misses simultaneously
  worsen. Do not claim distribution shift does not exist; claim only that
  reducing this measured generic shift metric did not improve performance
  under the tested intervention.
- Combined: the offline/online gap is better explained by the
  already-validated target-degeneracy finding (H3) than by either
  continuation-policy mismatch or generic state-distribution shift alone;
  H5/H6 are real but partial, secondary, regime-dependent contributors.

**Status (updated 2026-08-13): `SCIENTIFICALLY_COMPLETE_SYNTHESIS_PENDING`.**
Remaining work for this concern is manuscript/rebuttal synthesis only -- no
new local experiment is required or planned.

---

**Prior snapshot, retained for history (2026-08-12):** exact-target
replication, strict-preference/horizon, and learned/exact agreement are now
`FINAL_VALIDATED`. The two remaining local campaigns are C0/C1/C2
continuation causality and distribution-shift completion; their live
manifests, not this document, own progress counts.

- Concern paraphrase: does the mismatch between LRU-continuation label
  construction and learned-policy deployment (sequential distribution shift)
  explain some or all of the performance gap?
- Evidence complete locally: `PARTIAL` --
  `analysis/distribution_shift_ablation_v1/` covers `brightkite, citibike,
  cloudphysics` (3/7 families), 18/42 primary rows, as of the local
  checkpoint (superseded by the Wulver-merged 24/42 figure below).
  Trajectory divergence is large (97-99.8% at 3 capacities) but downstream
  misses *worsened* under the one directional test run (`DAGGER_ITER1`)
  despite a reduced measured state-shift index -- divergence exists, but
  the one causal-adjacent test does not show a simple fix.
- Known Wulver evidence (updated 2026-08-11, `WULVER_ONLY_VALIDATED`,
  relayed by the user):
  - Distribution-shift **merged state is now 24/42 rows** (up from the
    local 18/42 checkpoint). Across the 12 paired cells analyzed: measured
    state shift decreased in 9, misses improved in **zero**, misses
    worsened in **9**, misses tied in 3. This reinforces, and does not
    resolve, the existing negative finding -- do not claim distribution-
    shift correction solves the online-performance gap.
  - **The continuation-policy C0/C1/C2 full production campaign is now
    RUNNING locally**: launched 2026-08-11 in tmux session
    `kbs_continuation_c0_c1_c2_production_resume2_retry_20260812` (source SHA
    `a813617f36822f793b0e48b0ee3e6009d56ee324`), covering C0 LRU, C1 frozen
    `pi1`, and C2 trained from frozen-`pi1` continuation labels, with atomic
    unit completion, resume, same-example/leakage/model gates, and integrity
    outputs. No full scientific result exists yet -- do not cite an outcome
    until the 21-unit integrity manifest passes.
- Local sample-size evidence: the completed same-target learning curve
  narrows the causal space by disfavoring H1 within the tested `1%-50%`
  range, but it does **not** replace or solve Reviewer #3's missing
  C0/C1/C2 continuation experiment.
- Remaining experiment: monitor the running full 7-family C0/C1/C2
  production campaign to completion, then audit integrity before
  interpretation. Do not stop, signal, or relaunch it.
- Text-only fix: none required beyond what is already stated; the local docs
  already avoid claiming continuation mismatch is proven or that DAgger fixes
  the gap.
- **Status (2026-08-11 snapshot, superseded 2026-08-13 -- see top of this
  section): `PARTIAL`.** Primary missing experiment at the time: the true
  causal C0/C1/C2 continuation test -- this, not the LRB/3L-Cache/
  CACHEUS gap in Major 1, was the central unresolved issue for Reviewer #3's
  causal-explanation concern specifically. Both campaigns have since
  completed and passed formal integrity audit; see the 2026-08-13 snapshot
  above for the current, final status.

## Reviewer #2 Major 4: practical significance (computational cost)

- Concern paraphrase: is fine-grained candidate-level learned eviction
  computationally practical, or does its overhead undermine the contribution?
- Evidence complete locally: `SMOKE_ONLY` --
  `analysis/practical_significance_ablation_v1/exact_optimization_equivalence.json`
  shows `all_variants_exact_across_all_trace_capacity_pairs=true` with
  smoke speedups roughly `14.29x-99.99x`, but the artifact itself records
  `speedup_numbers_are_final_reviewer_evidence=false`.
- Known Wulver evidence (updated 2026-08-11, `WULVER_ONLY_VALIDATED`,
  relayed by the user): **the controlled timing campaign is now COMPLETE**,
  Wulver job `1171758`. Raw campaign 420/420 rows = 7 families x 3
  capacities x 4 policies x 5 repetitions. Audited mean per-request
  runtime: LRU `4.68us`, FIFO-Reinsertion `5.17us`, SIEVE `9.52us`,
  HALP-causal `870.66us` (~186x LRU in this implementation/protocol).
- Remaining experiment (updated 2026-08-13): none. The 420/420-row campaign
  has been synced from Wulver and independently re-audited locally (see
  `reports/kbs_final_evidence_20260813/controlled_timing_integrity.md`);
  policy means were independently recomputed from the raw file and match
  the pre-computed summary to 6 decimal places. Modern LRB/3L/CACHEUS timing
  is still not included in this 4-policy campaign and may still need a
  separate pass if required for a complete practical-significance table.
- Text-only fix: carry forward two caveats on promotion -- this is
  **wall-clock implementation evidence, not an algorithmic complexity
  theorem**, and the smoke-scale equivalence check remains a separate,
  still-valid supporting result (not superseded by the timing campaign).
- **Status (updated 2026-08-13): `EVIDENCE_SYNCED_SYNTHESIS_PENDING`.** The
  controlled timing campaign (420/420 rows, job `1171758`) has been synced
  from Wulver (13/13 transfer hashes PASS) and independently re-audited
  locally (structural gates PASS, policy means recomputed and matched).
  Controlled timing covers exactly LRU/FIFO-Reinsertion/SIEVE/HALP-causal, 5
  repetitions each; `evict_value_v1`'s runtime is a separate, single-run
  measurement recorded during its own held-out treatment run and must never
  be placed in this 4-policy table (see
  `reports/kbs_final_evidence_20260813/controlled_timing_interpretation.md`).
  Caveats unchanged: wall-clock implementation evidence only, and modern
  learned-baseline timing (LRB/3L/CACHEUS) may still be a separate item if
  needed.

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

**Authoritative reviewer-completion table (reconciled 2026-08-13, post manuscript/rebuttal synthesis):**

| Concern | Status | Primary remaining gap |
|---|---|---|
| R2 Major 1 (learned-baseline comparison) | `MANUSCRIPT_SYNTHESIS_COMPLETE` | Integrated as manuscript §3.6 (Table 7) and response-to-reviewers "Major Comment 1"; `evict_value_v1` loses on a clear majority of matched cells against every one of seven baselines including LRB/3L-Cache -- none |
| R2 Major 2 (supervision-objective ablation) | `MANUSCRIPT_SYNTHESIS_COMPLETE` | Integrated as manuscript §3.7 (Table 8); eviction-loss is worst/tied-worst of four objectives in every family -- none |
| R2 Major 3 (offline/online failure explanation) | `MANUSCRIPT_SYNTHESIS_COMPLETE` | Integrated as manuscript §3.8 (mechanistic diagnosis) and §3.9 (C0/C1/C2 + DAgger) -- none |
| R2 Major 4 (practical significance / timing) | `MANUSCRIPT_SYNTHESIS_COMPLETE` | Integrated as manuscript §3.10 (Table 9, controlled timing) and new §3.11 (Practical Significance) -- none |
| Reviewer #3 (causal explanation) | `MANUSCRIPT_SYNTHESIS_COMPLETE` | Integrated as manuscript §3.9; final answer `PARTIALLY_SUPPORTED` / `REGIME_DEPENDENT` -- none |

**RESPONSE_TO_REVIEWERS_COMPLETE**: `reports/kbs_response_to_reviewers_final.md` /
`submission_kbs_revision_final/02_Response_to_Reviewers.md` (+ regenerated
`.docx`) provide a full point-by-point response to all five concerns above,
with exact section/table/page references into the recompiled 44-page
manuscript PDF. Crosswalk: `reports/kbs_final_evidence_20260813/manuscript_reviewer_crosswalk.md`.
`NO_MAJOR_EXPERIMENT_REMAINS`. Next stage:
`FINAL_SUBMISSION_CONSISTENCY_AND_PACKAGE_AUDIT` (not yet performed -- do
not treat the submission package as upload-ready until that audit passes;
see `submission_kbs_revision_final/README_UPLOAD_CHECKLIST.md`).

- Most complete concerns: **Reviewer #2 Major 2** (supervision-objective
  ablation, `COMPLETE_VALIDATED`/`FINAL_VALIDATED` scientific scope) and, as
  of 2026-08-13, **Reviewer #2 Major 3 / Reviewer #3** (offline/online
  failure explanation and causal continuation-mismatch test) -- all planned
  local mechanistic science for these concerns is now complete and
  integrity-audited; remaining work everywhere in this row is manuscript
  integration, not further experimentation.
- Most under-addressed concern: **R3-Issue6 (fallback mechanism)** -- no
  local evidence exists at all, not even a smoke-scale implementation.
- `NO_NEW_EXPERIMENT_REQUIRED` locally for R2 Major 1-4 and Reviewer #3 as
  currently scoped. As of 2026-08-13, the two former Wulver-side
  synchronization blockers for R2 Major 1 and Major 4 (corrected
  `evict_value_v1` 42/42, controlled timing 420/420) have been synced and
  independently re-audited locally -- no synchronization or compute
  blockers remain anywhere in this table. Remaining work across all five
  rows is manuscript/rebuttal synthesis (`READY_FOR_MANUSCRIPT_AND_REBUTTAL_SYNTHESIS`).
