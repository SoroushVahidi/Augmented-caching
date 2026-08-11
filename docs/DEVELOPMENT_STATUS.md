# Development and Research Status

**Audience:** developers and future agents picking up this repository cold,
not reviewers and not a manuscript response document. If you are a new
agent with no conversation history, start here, then follow the checklist
in section 10.

**Last consolidated:** 2026-08-11 (local finalization/handoff pass, plus a
same-day reconciliation pass against fresh Wulver-side facts relayed by the
user -- see [`CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md`](CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md)
for the full detail behind every Wulver-sourced claim in this file). Live
run numbers in this file are timestamped snapshots -- see the convention
explained in [`kbs_second_revision_repository_state.md`](kbs_second_revision_repository_state.md)
and treat any fold/row count here as stale the moment time has passed.

---

## 1. Project purpose

This repository studies **learned cache-eviction policies**: instead of a
fixed heuristic (LRU, SIEVE, FIFO, ...), a model scores which resident
object to evict next, trained on a supervised target built from a
finite-horizon look-ahead over the request trace (`eviction_loss`, the
"exact number of future misses saved by evicting this candidate rather than
another, looking `H` requests ahead"). The core implementation is
`evict_value_v1` (`src/lafc/evict_value_*`), evaluated inside a general
cache-policy simulator (`src/lafc/simulator/`) against a library of
literature-faithful baselines (LRU family, marker/robust-combiner policies,
offline Belady, LRB, 3L-Cache, CACHEUS, a causal HALP reimplementation).

The project's central empirical finding, replicated across the public
`main` branch's evidence docs and this branch's deeper diagnostics, is a
**negative result under active investigation**: the learned model fits its
own supervision target well, but that target does not reliably translate
into good *online* decisions -- in the one deeply audited cell, exact
optimization of the finite-horizon target itself loses to plain LRU. The
work on this branch exists to find out *why*.

## 2. Current scientific question

**Why can finite-horizon eviction-loss supervision fail to translate into
good online cache behavior, even when the model fits that supervision
target almost exactly?**

This decomposes into two broad candidate failure classes, which the
hypothesis map (section 4 below, and the full
[`KBS_SECOND_REVISION_HYPOTHESIS_MAP.md`](reviewer/KBS_SECOND_REVISION_HYPOTHESIS_MAP.md))
tracks explicitly and separately:

- **Model-fit failure** -- the model does not approximate its own target
  well enough (H2, H7). Current evidence disfavors this as the primary
  explanation (96.5% exact-target agreement in the one audited cell).
- **Target-formulation failure** -- the target itself, even fit exactly,
  does not encode a good online decision rule, because it is degenerate
  (H3), truncated at too short a horizon (H4, H8, H9, H10, H11), or built
  under a continuation-policy assumption that does not match deployment
  (H5, H6). Current evidence points here as the stronger explanation, but
  every finding so far rests on **one or a few cells** (one trace family,
  one capacity, one horizon) and has **not been shown to generalize**.

## 3. Current branches

- **`main`** (public, GitHub `origin/main`) -- the stable, general-purpose
  entry point for outside readers. Contains: the simulator, `evict_value_v1`,
  the full literature-faithful baseline library including the validated LRB
  comparison, and five public evidence docs (`HYPOTHESIS_MAP.md`,
  `EXPERIMENT_REGISTRY.md`, `RESULTS_AND_LIMITATIONS.md`,
  `EXPERIMENTAL_EVIDENCE.md`, `REPRODUCIBILITY.md`). Everything documented
  as runnable on `main` must actually run on `main` today -- no aspirational
  claims. Historically, only `FINAL_VALIDATED` core-method and baseline work
  landed here; this finalization pass promotes a first slice of
  general-purpose mechanistic-diagnostic *source code* (see section 12).
- **`kbs/second-revision-science`** (this branch, research/reviewer-response
  line) -- the active scientific-development branch. Intended source of
  truth for reviewer-science code and frozen protocols; hosts the deeper
  mechanistic diagnostics (objective ablation, exact-target oracle, target
  degeneracy, distribution-shift/continuation-policy ablation, the learning
  curve campaign) before they are validated and, where general-purpose,
  promoted to `main`. Also hosts KBS-manuscript-specific and
  reviewer-numbered material that will likely **never** move to `main`
  (reviewer response drafts, per-reviewer coverage matrices, Wulver sync
  manifests).
- Other local worktrees (`Augmented-caching-3l-cache`, `-cacheus`,
  `-fairness`, `-halp`, `-kbs-parallel`, `-objective-ablation`) are
  feature-development or historical worktrees, not entry points -- see
  `git worktree list` for the current set.

## 4. Current experimental protocol

Canonical fairness protocol (`reviewer_fairness_v1`, shared via
`lafc.experiments.reviewer_fairness_common`):

- **Families:** 7 trace families -- `brightkite, citibike, cloudphysics,
  metacdn, metakv, twemcache, wiki2018`.
- **Capacities:** `{32, 64, 128}` object slots.
- **History window:** requests `[0, 10000)` (used to warm state / build
  features, not scored).
- **Score window:** requests `[10000, 50000)` -- 40,000 scored requests per
  cell.
- **Horizon:** `H = 4` requests for the primary `eviction_loss` target in
  all diagnostics run so far; longer horizons (`H = 8/16/32`) have been used
  only inside the target-degeneracy tie-break diagnostic, not as an
  alternative primary training target.
- **Main metrics:** `miss_ratio` (primary), plus per-diagnostic metrics
  (`validation_top1`, `validation_mean_regret`, `validation_pairwise_accuracy`,
  `mean_optimal_set_fraction`, `target_entropy_bits`, trajectory-divergence
  index, depending on the experiment).
- **Split principle:** leave-one-family-out cross-family folds
  (`cross_family_v1_<held_out_family>`); no held-out family's data is used
  to train a model evaluated on it.
- **Provenance principle:** every experiment output directory carries
  `provenance.json` (commit/branch/platform/python version/protocol scope)
  and, where relevant, `protocol_snapshot.json`; model artifacts are
  SHA-256 hashed; seed is fixed at `0` throughout the fairness-protocol
  line. See [`REPRODUCIBILITY.md`](../../Augmented-caching-main/docs/REPRODUCIBILITY.md)
  (main) and [`KBS_SECOND_REVISION_REPRODUCIBILITY.md`](reviewer/KBS_SECOND_REVISION_REPRODUCIBILITY.md)
  (this branch) for full mechanics.

## 5. Completed evidence

Status vocabulary used below (canonical on this branch, see the registry):
`FINAL_VALIDATED`, `COMPLETE_DIAGNOSTIC`, `COMPLETE_PARTIAL_SCOPE`,
`EXPERIMENTALLY_COMPLETE_SYNTHESIS_PENDING`, `RUNNING`, `PENDING`,
`IMPLEMENTATION_READY`, `SMOKE_ONLY`, `BLOCKED`, `SUPERSEDED`,
`INVALID_DO_NOT_USE`. Full detail and per-row provenance:
[`KBS_SECOND_REVISION_EXPERIMENT_REGISTRY.md`](reviewer/KBS_SECOND_REVISION_EXPERIMENT_REGISTRY.md)
(this branch, canonical) / [`EXPERIMENT_REGISTRY.md`](../../Augmented-caching-main/docs/EXPERIMENT_REGISTRY.md)
(main, public-safe subset).

| Experiment | Scope | Status | Main finding | Artifact / config | Confidence |
|---|---|---|---|---|---|
| Objective ablation (`eviction_loss` vs `next_arrival` vs `reuse_distance` vs `pairwise`) | 7 families x 4 objectives x 3 capacities | `FINAL_VALIDATED` | `eviction_loss` is worst or tied-worst in every one of 7 families | `analysis/supervision_objective_ablation_v1/`, `configs/supervision_objective_ablation_protocol_v1.json` | High (full scope) |
| Corrected held-out `evict_value_v1` cross-family replay | Fairness protocol | `COMPLETE_PARTIAL_SCOPE` | Not yet a usable primary comparison table; top-priority open item | `analysis/reviewer_fairness_cross_family_v1/` | Low (incomplete) |
| Exact-protocol LRU / SIEVE / FIFO | Full protocol | `FINAL_VALIDATED` | Primary baseline evidence, controlled window | `analysis/reviewer_fairness/` (`primary_controlled_window` rows only) | High |
| Learned-baseline comparison (LRB / 3L-Cache / CACHEUS / causal-HALP) | Full protocol | Baseline side `FINAL_VALIDATED` (with fidelity caveats); R2 Major 1 overall `EXPERIMENTALLY_COMPLETE_SYNTHESIS_PENDING` | Primary baseline evidence; LRB is `LOCAL_EXACT_PROTOCOL_VALIDATED`, 3L-Cache is `LOCAL_EXACT_PROTOCOL_VALIDATED_WITH_CAVEAT`, CACHEUS is `LOCAL_EXACT_PROTOCOL_VALIDATED_WITH_PROVENANCE_CAVEAT`; final treatment-vs-baseline synthesis awaits local sync of the Wulver-only corrected `evict_value_v1` CSV | `analysis/reviewer_fairness/`, `analysis/kbs_r2_major1_evidence_prep_20260811/` | High for baseline side; synthesis pending for treatment side |
| Exact-target oracle diagnostic | 1 cell (brightkite, cap 64, H=4) | `COMPLETE_DIAGNOSTIC` | Learned model beats exact target-oracle optimization (15,449 vs 19,079 misses); both worse than LRU (13,225); Belady 11,312 | `analysis/exact_target_oracle_diagnostic_v1/brightkite_cap64_h4/` | High for this cell only; not family-general |
| Target-degeneracy / horizon tie-resolution diagnostic | 1 cell | `COMPLETE_DIAGNOSTIC` | Strongest-supported mechanistic finding on this branch: 99.3% of candidates tie for "optimal" under H=4; 8x horizon only breaks a minority of ties | `analysis/eviction_loss_target_degeneracy_v1/brightkite_cap64_h4/` | High for this cell only; not family-general |
| Historical-tail diagnostic | -- | `BLOCKED` locally (`NOT_LOCAL`) | Not implemented locally; ownership per older local docs is Wulver-side, **not independently reverified this pass** | n/a | n/a -- see `LAST_KNOWN_REMOTE_STATUS` caveat below |
| Distribution-shift / continuation-policy preliminary check | 1 family (metacdn) | `COMPLETE_PARTIAL_SCOPE` | Trajectory divergence real (97-99.8% at 3 capacities); DAgger relabeling made misses *worse*, not better | `analysis/distribution_shift_ablation_v1/` | Medium; single family |
| Continuation-policy causal ablation (C0/C1/C2, frozen protocol) | -- | `PRODUCTION_RUNNER_READY_SMOKE_VALIDATED` | Resumable production runner now exists and has been smoke-validated on one `(family, capacity)` unit with C0/C1/C2 rows, label agreement, pi2 training, atomic unit manifest, provenance, integrity checks, and resume skip/no-duplicate behavior. No scientific result yet; full campaign is not launched. The learning-curve closeout does not solve this Reviewer #3 item. | `src/lafc/continuation_policy_ablation.py`, `scripts/experiments/run_continuation_policy_causal_ablation.py`, `configs/continuation_policy_causal_ablation_production_v1.json` | None yet (not run at scale) |
| Continuation-policy light ablation (earlier, historical) | 4 traces | `SUPERSEDED` by C1/C2 above | Kept for provenance only; do not conflate with the frozen C1/C2 protocol | -- | n/a -- superseded |
| Learning convergence (scalar vs pairwise, same target) | Fraction sweep `1%, 2%, 5%, 10%, 25%, 50%` | `FINAL_50PCT_VALIDATED` | `50%`: 7/7 families, 42/42 rows, all `status=ok`, duplicate-key count 0, NaN/Inf count 0, 30 audit files total, 7/7 fraction-0.5 audit units, 0 model SHA mismatches. Apples-to-apples `1%->50%` shows no material monotonic downstream improvement; pairwise remains flat/worse. Stopping decision: `STOP_SAMPLE_SIZE_HYPOTHESIS`; `100%` intentionally not run. | `analysis/supervision_objective_learning_curve_v1/`, `analysis/supervision_objective_learning_curve_v1/final_50pct_synthesis_20260811/`, `configs/supervision_objective_learning_curve_v1.json` | High for intended stopping-rule scope |
| Practical significance / controlled timing | -- | Equivalence check `COMPLETE_DIAGNOSTIC` (`SMOKE_ONLY` for timing numbers) | A vectorized reimplementation makes identical decisions to the reference implementation in checked cells; final controlled timing campaign not yet run | `analysis/practical_significance_ablation_v1/` | High for equivalence; timing numbers not citable as final |
| Cross-cutting comparison-fairness audit | -- | `FINAL_VALIDATED` (as an audit) | Overall fairness score 76/100, `GENERALLY_FAIR_WITH_LIMITATIONS`; the historical `evict_value_v1` loss to LRU/SIEVE/FIFO is unlikely explained by protocol unfairness (the one confirmed unfairness, train/test overlap, would have *favored* `evict_value_v1`, yet it still lost) | `docs/reviewer/kbs_comparison_fairness_audit.md` | High as an audit of methodology, not a performance claim |
| Distribution-shift (Wulver-merged 24/42) | -- | `WULVER_ONLY_VALIDATED` | 24/42 rows (up from the local 18/42 checkpoint). Across 12 paired cells: measured state shift decreased in 9, misses improved in **0**, misses worsened in **9**, misses tied in 3 -- reinforces, does not resolve, the existing negative-result narrative | Not locally present | See `CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md` |
| Broad target-degeneracy (21 cells, 7 families x 3 capacities) | -- | `WULVER_ONLY_VALIDATED` | Wulver job `1169513`. Unique-winner fraction = 0 and multi-winner fraction = 1 across **all 21 cells** -- generalizes the local single-cell H3 finding well beyond one cell. Capacity trend (zero-margin fraction and optimal-set fraction both rise with capacity) is **empirical evidence, not a mathematical H/C law** | Not locally present | See `CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md` |
| Horizon sensitivity sweep | -- | `RUNNING` (Wulver-side) | Wulver job `1169299`, **17/35 complete**, 18 pending. H=1 and H=2 complete for all families; H=4 complete for brightkite/citibike/cloudphysics only; remaining H=4 cells and all H=8/H=16 pending | Not locally present | See `CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md` |
| Historical-tail diagnostic (H8) | -- | `WULVER_ONLY_VALIDATED` | Wulver job `1169665`, complete. H=8 resolves ~24.6% of H=4-tied decisions; H=16 resolves ~38.7%; history-linear tie-breaking produces only tiny gains; leakage audit passed. Weak support for horizon/tail concerns, **not a downstream policy win** | Not locally present, not previously implemented locally at all | See `CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md` |
| Reuse-tail horizon diagnostic (`P(T > H | resident)`) | -- | `LOCAL_COMPLETE` | 21/21 family-capacity cells, capacities `32/64/128`, horizons `1/2/4/8/16`, integrity passed. H=4: `P(T>4 | resident)=0.9938544459677984`, `P(T>4 | resident, eventually reused)=0.9793302186526528`, never-reused fraction `0.7026792916224847`. Supports horizon observability limits, **not causal excess-miss attribution** | `analysis/reuse_tail_horizon_diagnostic_v1/`; synthesis in `docs/reuse_tail_horizon_diagnostic_v1_synthesis.md` | No rerun needed |
| Corrected held-out `evict_value_v1` (cross-family, 42/42) | -- | `WULVER_ONLY_VALIDATED`; local classification `FINAL_COMPARISON_PENDING` | Complete on Wulver per relayed audit: 42/42 rows, 7 families x 3 capacities x 2 variants, all `ok`, SHA-256 `982bfdffdbd816b56c2eef86ecb730a1eb136b3f85e36ad533739e586fa0a296`. Comprehensive local path/name/SHA search on 2026-08-11 found no local copy; old contaminated local `policy_comparison_evict_value_v1.csv` remains excluded | `analysis/reviewer_fairness_cross_family_v1/evict_value_v1_final_42_20260810/policy_comparison.csv` (Wulver path; not yet synced locally); status manifest in `analysis/kbs_r2_major1_evidence_prep_20260811/treatment_status.json` | Sync and verify this one CSV/provenance, then run `scripts/analysis/prepare_r2_major1_evidence.py --treatment-csv ...` |
| Exact controlled-window LRB / 3L-Cache / CACHEUS | -- | `LOCAL_EXACT_PROTOCOL_VALIDATED` / `LOCAL_EXACT_PROTOCOL_VALIDATED_WITH_CAVEAT` / `LOCAL_EXACT_PROTOCOL_VALIDATED_WITH_PROVENANCE_CAVEAT` plus `WULVER_PENDING` replication | Fresh 2026-08-11 local audit: each CSV has 42 rows, 21 primary `[10000,50000)` rows, all seven families, capacities `32/64/128`, all `ok`, no duplicate keys, no NaN/Inf, and matching SHA-256. Trace manifest gap repaired by hashing the seven local 50k trace files. Wulver jobs `1171965` (3L-Cache), `1171966` (LRB), `1171967` (CACHEUS) remain pending because of maintenance, but are replication/provenance strengthening rather than required experiments unless their missing config later proves materially different | `analysis/reviewer_fairness/`; compact evidence package `analysis/kbs_r2_major1_evidence_prep_20260811/`; script `scripts/analysis/prepare_r2_major1_evidence.py` | No local rerun needed; Wulver copy is optional replication unless config differs |
| Controlled timing campaign | -- | `WULVER_ONLY_VALIDATED` | Wulver job `1171758`, complete. 420/420 rows (7 families x 3 capacities x 4 policies x 5 repetitions). Mean per-request runtime: LRU 4.68us, FIFO-Reinsertion 5.17us, SIEVE 9.52us, HALP-causal 870.66us (~186x LRU) | Not locally present | `PROMOTE_NOW`, see `WULVER_TO_GITHUB_PROMOTION_QUEUE.md` #2 |
| Continuation-policy causal ablation (C0/C1/C2) | -- | `PRODUCTION_RUNNER_READY_SMOKE_VALIDATED` | Production runner added locally with atomic `(held_out_family, capacity)` units, fail-closed preflight, same-example/leakage/model gates, and resume. Tiny runner smoke and resume smoke passed. **Not complete and not running** until the real 21-unit campaign is explicitly launched. | n/a | `READY_FOR_LOCAL_LAUNCH_DECISION`, see `WULVER_TO_GITHUB_PROMOTION_QUEUE.md` #9 |

For remote-only/Wulver-adjacent evidence: **this workstation still does not
contact Wulver, SSH to Wulver, or query Slurm directly.** Every row above
labeled `WULVER_ONLY_VALIDATED` or `PENDING`/`RUNNING` (Wulver-side) was
relayed by the user from a separately audited Wulver-side session on
2026-08-11, not independently verified by this workstation -- see the
provenance note at the top of
[`CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md`](CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md).
This supersedes the previous, more conservative `LAST_KNOWN_REMOTE_STATUS --
NOT RECHECKED IN THIS PASS` framing for the specific items listed above.
The reuse-tail `P(T > H | resident)` diagnostic is now explicitly listed as
`LOCAL_COMPLETE`; items not listed above still need fresh local inspection
before they are treated as implemented.

## 6. Invalid / superseded evidence

Do not use these for final claims:

| Artifact | Why invalid | Status |
|---|---|---|
| `analysis/reviewer_fairness/policy_comparison_evict_value_v1.csv` | Confirmed train/test overlap (`split_mode=trace_chunk` reused the same 7 canonical trace streams for train and eval); only 5 of 21 expected primary rows present | `CONTAMINATED_DO_NOT_USE` -- never cite |
| `deployment_full_stream` rows in any fairness-protocol CSV | Not the primary controlled-window comparison; supporting context only, not eligible for the primary reviewer table | Exclude from primary comparisons |
| Continuation-policy light ablation (pre-C1/C2, 4 traces, <=300 requests/trace) | `SUPERSEDED` by the frozen C1/C2 causal-ablation protocol | Kept for provenance only |
| Practical-significance timing numbers | Explicitly recorded `speedup_numbers_are_final_reviewer_evidence=false` in their own artifact | Must always carry `SMOKE_ONLY` qualifier when cited |
| `objective_pairwise` results cited as equivalent to `eviction_loss_pairwise` results | Naming trap -- `objective_pairwise` changes the supervision objective; `eviction_loss_pairwise` keeps the eviction-loss target fixed and only changes representation. These are **not interchangeable** | Always check the exact `condition` column value before citing |
| Historical `*_heavy_r1` manuscript-pipeline material (`docs/wulver_heavy_evict_value_experiment.md`, `docs/evict_value_v1_kbs_canonical_artifacts.md`, etc.) | Belongs to the original (pre-second-revision) KBS submission pipeline, not this branch's diagnostic line | `HISTORICAL` -- valid provenance, not default source of truth here |
| `offline_belady` cited as a deployable baseline | It is a future-aware oracle, never a deployable policy | Always label as oracle/upper-bound context |
| `docs/manuscript_open_questions.md`, `docs/manuscript_evidence_map.md` | Last touched 2026-04-10/11, ~4 months before this branch's current work; reference an earlier TIST-framing pairwise-vs-pointwise research line and do not mention `evict_value_v1`, the H1-H11 hypothesis map, or the second-revision experiment registry at all | `LIKELY_SUPERSEDED / ORPHANED` -- not cross-referenced by any current second-revision doc; flagged here rather than deleted (do not delete historical evidence) |
| Any header-only or failed held-out evaluation output | Not usable scientific evidence | Historical failure provenance only |

## 7. Completed local learning-curve closeout

**Confirmed via direct inspection at 2026-08-11 after the final
`wiki2018|0.5` resume completed naturally.** This closeout was
documentation/synthesis-only: no new experiment was launched, no 100%
fraction was run, no Wulver contact occurred, and no tmux session was
stopped or signaled.

- Fraction `0.5` is complete: 7/7 families, 42/42 rows, all `status=ok`,
  capacities `32/64/128`, both conditions (`eviction_loss_scalar`,
  `eviction_loss_pairwise`), duplicate-key count 0, NaN/Inf count 0.
- Artifact integrity passed: expected 50% model files are present, 0 model
  SHA mismatches, 30 audit files total, 7/7 fraction-0.5 audit units, and
  `campaign_state.json` contains all seven `family|0.500000` units including
  `wiki2018|0.500000`.
- Synthesis path:
  `analysis/supervision_objective_learning_curve_v1/final_50pct_synthesis_20260811/`.
- Apples-to-apples learning-curve summary over the four families present at
  every fraction (`brightkite`, `citibike`, `cloudphysics`, `metacdn`):
  scalar MAE / scalar miss_ratio / pairwise miss_ratio:
  `1%`: 0.9867 / 0.6256 / 0.8299; `10%`: 0.9825 / 0.6110 / 0.8297;
  `25%`: 0.9826 / 0.6137 / 0.8296; `50%`: 0.9827 / 0.6126 / 0.8300.
  The `1%->50%` curve does not show material monotonic downstream
  improvement.
- Full 50% seven-family comparison: scalar better on 18/21
  family/capacity cells, ties on 3/21, pairwise better on 0/21; mean
  pairwise-minus-scalar miss-ratio gap is approximately `+0.1611`.
- Scientific conclusion for H1: `STOP_SAMPLE_SIZE_HYPOTHESIS`. Within the
  tested `1%-50%` range, the sample-size explanation is not supported as
  the primary cause. This is not a claim that more data can never help;
  it is the campaign's predefined stopping-rule conclusion.

## 8. Open experiments (ranked by priority)

See [`NEXT_STEPS.md`](NEXT_STEPS.md) for the full P0-P4 roadmap and
[`WULVER_TO_GITHUB_PROMOTION_QUEUE.md`](WULVER_TO_GITHUB_PROMOTION_QUEUE.md)
for the Wulver-side sync/promotion priority order. Summary ranking, updated
for the 2026-08-11 Wulver reconciliation (two items below are now
sync/review tasks rather than experiments to run, since Wulver already
completed them):

1. Sync and review the corrected held-out cross-family `evict_value_v1` replay (42/42, Wulver-complete) -- now the only R2 Major 1 synthesis blocker, not a re-run.
2. Use the locally validated exact controlled-window LRB/3L-Cache/CACHEUS CSVs as the baseline side of that table; rerun `scripts/analysis/prepare_r2_major1_evidence.py --treatment-csv ...` after the verified treatment CSV is synchronized. Sync Wulver jobs `1171965`-`1171967` later only for replication or if their missing config records a material difference.
3. Decide whether to launch the continuation-policy C0/C1/C2 production runner locally -- runner/preflight/smoke/resume are ready, but the full campaign itself remains unlaunched.
4. Complete the target-degeneracy and exact-target-oracle diagnostics across all 7 families / 3 capacities locally (currently 1 cell each on this workstation; Wulver has already generalized target-degeneracy to 21 cells -- see item 3 in `WULVER_TO_GITHUB_PROMOTION_QUEUE.md` for syncing that instead of re-running it locally).
5. No-op closeout for the `P(T > H | resident)` reuse-time-tail diagnostic: it is now locally complete and synthesized; only causal excess-miss attribution remains unimplemented.
6. Sync the controlled timing campaign (420/420, Wulver-complete, `PROMOTE_NOW`) -- no local work needed, just sync.
7. Sync the exact-protocol LRB/3L-Cache/CACHEUS Wulver jobs (`1171965`-`1171967`) later only for independent replication or if their missing config records a material difference from the locally audited controlled-window CSVs.

## 9. Current scientific interpretation

- **Target-formulation problem is currently the strongest explanation, and
  is no longer resting on a single cell.** H3 (target degeneracy) is
  `STRONGLY_SUPPORTED` in the original locally-audited cell (99.3% of
  candidates tie for "optimal" under H=4), and per the Wulver-relayed
  21-cell broad degeneracy result (job `1169513`, see
  `CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md`), the unique-winner fraction is
  **zero across all 21 family/capacity cells** -- this generalizes the
  local single-cell finding substantially, though it is not yet
  independently reproduced on this workstation. The Wulver-relayed
  historical-tail result (job `1169665`: H=8/H=16 resolve only ~24.6%/38.7%
  of H=4 ties) provides *weak* additional support for the horizon/tail
  side of this story specifically -- explicitly not a downstream policy
  win, only a tie-resolution measurement.
- **Insufficient-data / pure model-fit explanations are currently
  disfavored.** H1 (more data would close the gap) shows flat offline and
  downstream metrics from 1% to 50% of training data, and the predefined
  stopping decision is `STOP_SAMPLE_SIZE_HYPOTHESIS`. H2 (model fails to
  fit its target) is contradicted by 96.5% exact-target agreement with low
  mean regret in the one audited cell -- the model's small departures from
  its own target are net beneficial, not harmful.
- **Continuation-policy mismatch (H5/H6) is unresolved, not disfavored --
  and the Wulver-merged distribution-shift state reinforces caution rather
  than resolving anything.** The Wulver-merged 24/42 distribution-shift
  state (up from the local 18/42 checkpoint) analyzed 12 paired cells:
  measured state shift decreased in 9 of them, but misses improved in
  **zero** of those 9 and worsened in 9 of the 12 overall. This is
  consistent with, but does not prove, a continuation-mismatch mechanism --
  do not read it as evidence that fixing distribution shift would fix the
  gap. The properly causal C0/C1/C2 test remains the decisive experiment for
  H5. The local production runner is now
  `PRODUCTION_RUNNER_READY_SMOKE_VALIDATED`; the full 21-unit campaign has
  not been launched.
- **Horizon/truncation mechanism (H4, H8-H11) is now supported as an
  observability limitation, but not as a causal excess-miss mechanism.** The
  local reuse-tail diagnostic shows that the H=4 window sees very little of
  resident objects' eventual reuse behavior:
  `P(T>4 | resident)=0.9938544459677984`, and even among objects eventually
  reused, `P(T>4 | resident, eventually reused)=0.9793302186526528`.
  The trend worsens with capacity (`0.987377360773`, `0.992891528854`,
  `0.996078682684` for C=32/64/128), directionally matching the
  Wulver-relayed broad H=4 degeneracy trend. This strengthens H4/H10/H11
  conservatively, but it remains potential unseen future reuse, not proof
  that reuse after H caused an avoidable miss.
- **No causal claims are established yet.** Every finding above is
  correlational/diagnostic on one or a few cells. In particular: `P(T>H)`
  being large would be *consistent with* a truncation explanation but is
  not itself proof of a causal excess-miss mechanism; that requires the
  not-yet-implemented counterfactual-replay mechanism sketched under H11.

## 10. Next-agent startup checklist

1. Read this file (`docs/DEVELOPMENT_STATUS.md`) in full.
2. Read the public evidence docs on `main`: `HYPOTHESIS_MAP.md`,
   `EXPERIMENT_REGISTRY.md`, `RESULTS_AND_LIMITATIONS.md`,
   `EXPERIMENTAL_EVIDENCE.md`, `REPRODUCIBILITY.md`.
3. Inspect current git status in whichever worktree(s) you're using
   (`git status --short`, `git log --oneline -15`, `git fetch origin` then
   compare `HEAD` against `origin/<branch>`) -- do not trust this file's
   git-state numbers once time has passed.
4. Inspect tmux (`tmux ls`) and the process table before assuming any
   long-running experiment has finished or is safe to touch.
5. Do not overwrite active or generated outputs under `analysis/`, `models/`,
   or `logs/` -- these are evidence, not scratch space, even when
   gitignored.
6. Do not contact Wulver, SSH to Wulver, or query Slurm unless the current
   task explicitly authorizes it -- treat all `LAST_KNOWN_REMOTE_STATUS`
   labels in this file as unverified until rechecked.
7. Continue the highest-priority incomplete item from section 8 /
   `NEXT_STEPS.md`, respecting whatever is currently running locally.

## 11. Known cross-machine synchronization state

Sourced entirely from existing local sync docs (not rechecked against
Wulver this pass):
[`KBS_LOCAL_TO_WULVER_MASTER_MANIFEST.md`](reviewer/KBS_LOCAL_TO_WULVER_MASTER_MANIFEST.md),
[`KBS_LOCAL_TO_WULVER_SYNC_STATUS.md`](reviewer/KBS_LOCAL_TO_WULVER_SYNC_STATUS.md),
[`KBS_LOCAL_WULVER_CONFLICT_MATRIX.md`](reviewer/KBS_LOCAL_WULVER_CONFLICT_MATRIX.md),
[`local_to_wulver_continuation_sync_manifest.md`](reviewer/local_to_wulver_continuation_sync_manifest.md).

`LAST_KNOWN_REMOTE_STATUS -- NOT RECHECKED IN THIS PASS`:

- These docs assert (as of their own last local edit, 2026-08-10) that
  several files are believed to exist only on Wulver and not yet synced
  back locally: `scripts/experiments/run_distribution_shift_family.py`,
  `scripts/experiments/upgrade_cross_family_manifest_metadata.py`, and four
  `slurm/kbs_*.sbatch` drivers (distribution-shift and cross-family-held-out
  smoke/full variants).
- `src/lafc/supervision_objective_ablation.py` was flagged
  `NEEDS_SEMANTIC_REVIEW` for cross-machine merge purposes (largest shared
  diff, +200/-63 lines relative to what these docs believed was the
  Wulver-side state). This pass's local-only comparison (main vs. this
  worktree) found the file **does not exist on `main` at all** -- see
  section 12 -- so the "shared kernel conflict" risk these docs anticipated
  is, locally, actually a clean promotion opportunity rather than a merge
  conflict. Whether Wulver independently modified this file is still
  unverified.
- Generated evidence now eligible for intentional transfer after audit: the
  completed 50% learning-curve output tree and final synthesis (section 7),
  the one-cell oracle/degeneracy diagnostics, and the metacdn
  distribution-shift partial checkpoint.
- Do not treat any of the above as current fact; re-verify by actually
  contacting Wulver in a session explicitly authorized to do so.
- **2026-08-11 update:** several of the open items above have since been
  addressed via user-relayed facts from a separately-audited Wulver
  session (not independent verification by this workstation) -- see
  section 5's Wulver-sourced rows and
  [`CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md`](CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md)
  for the full detail. In particular, the file-existence gap flagged above
  (`run_distribution_shift_family.py`,
  `upgrade_cross_family_manifest_metadata.py`, four `slurm/kbs_*.sbatch`
  drivers) was **not** resolved by this update and remains open -- the new
  facts are about experiment *results*, not about locating these missing
  orchestration files.

## 12. Git state / pending consolidation

As of this pass (2026-08-10 23:48 EDT, after `git fetch origin` in both
worktrees):

- `kbs/second-revision-science` local `HEAD` was **30 commits ahead** of
  `origin/kbs/second-revision-science`, working tree clean, before this
  pass's own commits (see the final git-push section of the handoff report
  for the post-push state).
- `main` local `HEAD` matched `origin/main` exactly (0 ahead / 0 behind),
  working tree clean.
- **What exists only on `kbs/second-revision-science`:** the entire
  mechanistic-diagnostic source surface -- `src/lafc/supervision_objective_ablation.py`
  (the shared candidate-row-building kernel used by every diagnostic below),
  `src/lafc/target_degeneracy.py`, `src/lafc/oracle_diagnostics.py`,
  `src/lafc/continuation_policy_ablation.py`,
  `src/lafc/supervision_objective_ablation_train.py`, `src/lafc/halp_model.py`,
  the learning-curve campaign runner
  (`scripts/experiments/run_supervision_objective_learning_curve.py`), all
  associated tests and frozen configs, plus every reviewer-facing doc under
  `docs/reviewer/` and the KBS-specific status/sync docs at the repo root of
  `docs/`. 88 files differ between `main` and this branch under
  `src/`, `tests/`, `configs/` alone (`+12,430/-7` lines), across 68
  commits not on `main`.
- **What exists only on `main`:** nothing structurally significant beyond
  what this branch already has a superset of; `main` is a strict ancestor
  state for the `src/`/`tests/`/`configs/` surface relevant to this
  investigation (this branch was created from `main`'s history and has only
  added, not removed, from that surface).
- **Promotion audit result (this pass):** a real dependency-graph analysis
  (imports, test portability, shared-kernel risk) found the
  `supervision_objective_ablation.py` kernel is self-contained, imports
  cleanly against `main`'s existing modules, and is not exercised by any
  existing `main` test -- i.e., promoting it carries no compatibility risk
  to current `main` experiments. See section 10 of the top-level task
  report / the commit history on `main` for exactly what was promoted in
  this pass and what was deliberately deferred (the continuation-policy
  chain, the learning-curve runner, and the two independently-evolved
  `scripts/revision_status.py` tools, which have a genuine naming/logic
  collision requiring a deliberate human merge decision, not a copy).
