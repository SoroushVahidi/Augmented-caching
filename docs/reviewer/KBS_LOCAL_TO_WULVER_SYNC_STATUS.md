# KBS Local To Wulver Sync Status

Date: 2026-08-09

Scope: local repository audit only. No Wulver contact was performed during this
pass, so Wulver runtime state is intentionally not asserted here.

For the mechanistic-hypothesis matrix see
[`KBS_SECOND_REVISION_HYPOTHESIS_MAP.md`](KBS_SECOND_REVISION_HYPOTHESIS_MAP.md);
for the per-reviewer-concern status matrix see
[`KBS_SECOND_REVISION_REVIEWER_COVERAGE.md`](KBS_SECOND_REVISION_REVIEWER_COVERAGE.md).
This file stays focused on what is/isn't synced between local and Wulver.

## A. Already On Both / No Sync Needed

These are believed to be common baseline branch material or already represented
in the tracked local history:

- core `lafc` package and simulator infrastructure
- frozen fair-cross-family fold configs under `configs/fair_cross_family_v1/`
- reviewer fairness protocols and common result schema
- supervision-objective ablation code, registry gates, training wrappers, and
  held-out evaluation wrappers already in local branch history
- existing model registry and objective-ablation artifacts preserved locally
  under ignored `analysis/supervision_objective_ablation_v1/` and
  `models/supervision_objective_ablation_v1/`

Verify against Wulver before launch; do not assume this list is a substitute
for a file-level remote diff.

## B. Local-Only Source / Config / Test Work

These need local -> Wulver synchronization before Wulver can run or reproduce
the matching diagnostics:

Continuation-policy causal ablation:

- `src/lafc/continuation_policy_ablation.py`
- `scripts/experiments/run_continuation_policy_causal_ablation_smoke.py`
- `tests/test_continuation_policy_ablation.py`
- `configs/continuation_policy_causal_ablation_v1.json`
- `docs/reviewer/local_to_wulver_continuation_sync_manifest.md`

Exact-target oracle foundation:

- `src/lafc/oracle_diagnostics.py`
- `scripts/experiments/run_exact_target_oracle_diagnostic.py`
- `tests/test_oracle_diagnostics.py`
- shared helper changes in `src/lafc/supervision_objective_ablation.py`

Target-degeneracy diagnostic:

- `src/lafc/target_degeneracy.py`
- `scripts/experiments/analyze_eviction_loss_target_degeneracy.py`
- `tests/test_target_degeneracy.py`
- shared helper changes in `src/lafc/supervision_objective_ablation.py`

Learning-convergence tooling:

- `scripts/experiments/run_supervision_objective_learning_curve.py`
- `configs/supervision_objective_learning_curve_v1.json`
- `tests/test_supervision_objective_learning_curve.py`
- supporting reviewer diagnostics in `src/lafc/reviewer_diagnostics.py`

Reviewer/status documentation:

- `docs/kbs_second_revision_repository_state.md`
- `docs/reviewer_revision_roadmap.md`
- `docs/reviewer/kbs_second_revision_artifact_map.md`
- `docs/reviewer/kbs_negative_results_interpretation.md`
- `docs/reviewer/kbs_comparison_fairness_audit.md`
- this sync inventory

## C. Local-Only Generated Results

These are generated evidence/preservation artifacts. They should not be mixed
with source commits casually, and they should not be blindly copied into a
source-only sync package.

Preserve locally and later consolidate intentionally:

- `analysis/supervision_objective_learning_curve_v1/`
  - local diagnostic output; `25%` phase completed naturally `2026-08-09`
    (as of the `2026-08-10` read-only audit): all `7/7` families
    (`brightkite`, `citibike`, `cloudphysics`, `metacdn`, `metakv`,
    `twemcache`, `wiki2018`), `42/42` rows, integrity verified. `50%`
    first attempt `INTERRUPTED_BEFORE_FIRST_COMPLETED_UNIT` (launched
    `2026-08-10` outside tmux in a foreground SSH shell, terminated after
    ~80 minutes when the SSH session closed, `0/42` rows committed);
    relaunched same day in tmux session
    `kbs_learning_curve_50pct_20260810`, now `RUNNING_LOCAL_RESUME` and
    confirmed healthy; `4/7` folds complete as of a `2026-08-10 21:59`
    read-only re-check (`brightkite`, `citibike`, `cloudphysics`,
    `metacdn`; `metakv` in progress; `twemcache`, `wiki2018` remaining;
    a further resume may be needed if the `10`-hour clean-stop is hit
    first); `100%` not yet started.
- `models/supervision_objective_learning_curve_v1/`
  - generated models for completed learning-curve units
- `analysis/exact_target_oracle_diagnostic_v1/brightkite_cap64_h4/`
  - completed one-cell exact-target oracle diagnostic
- `analysis/eviction_loss_target_degeneracy_v1/brightkite_cap64_h4/`
  - completed one-cell target-degeneracy diagnostic
- `analysis/kbs_comparison_fairness_audit.json`
  - generated local fairness audit summary

Existing ignored generated evidence that should continue to be preserved:

- `analysis/reviewer_fairness/`
- `analysis/reviewer_fairness_cross_family_v1/`
- `analysis/distribution_shift_ablation_v1/`
- `analysis/practical_significance_ablation_v1/`
- `analysis/supervision_objective_ablation_v1/`
- `analysis/external_learned_baselines/`
- `models/`
- `logs/`

## D. Wulver-Only Known Work

Known from recorded local status, requiring later Wulver -> local
consolidation. This audit cannot verify latest Wulver state.

- corrected held-out orchestration/results
- horizon-sensitivity orchestration/results
- Wulver status/inventory work
- Wulver-only distribution-shift family runner and Slurm launchers, including
  previously recorded missing local files:
  - `scripts/experiments/run_distribution_shift_family.py`
  - `scripts/experiments/upgrade_cross_family_manifest_metadata.py`
  - `slurm/kbs_distribution_shift_wulver_smoke.sbatch`
  - `slurm/kbs_distribution_shift_wulver.sbatch`
  - `slurm/kbs_cross_family_heldout_smoke.sbatch`
  - `slurm/kbs_cross_family_heldout_eval_wulver.sbatch`

## E. Conflict-Risk Files

These require manual/semantic merge rather than blind overwrite because they
are likely to have changed independently on local and Wulver or are central
status files:

- `docs/reviewer_revision_roadmap.md`
- `docs/kbs_second_revision_repository_state.md`
- `docs/reviewer/kbs_second_revision_artifact_map.md`
- `docs/reviewer/kbs_negative_results_interpretation.md`
- `configs/reviewer_revision_roadmap.json`
- `scripts/experiments/run_distribution_shift_ablation.py`
- `scripts/experiments/run_evict_cross_family_pipeline.py`
- `scripts/experiments/run_evict_value_v1_cross_family_eval.py`
- `configs/fair_cross_family_v1/folds/*.json`
- any `slurm/kbs_*` launch file
- any `analysis/*/provenance.json` copied from either side

Safe transfer strategy:

- sync source/config/test files through a branch or patch series, not by
  overwriting the Wulver worktree;
- copy generated result directories only into clearly named preservation paths;
- compare hashes and provenance before treating copied result files as final;
- merge documentation by topic, preserving status labels:
  `OBSERVED`, `SUPPORTED HYPOTHESIS`, `OPEN QUESTION`, `RUNNING`, `NOT RUN`,
  and `SMOKE ONLY`.

## F. Current Theory / Work-Map Boundaries

Keep these mechanisms separate during any later sync review:

- target-formulation branch:
  short-H eviction-loss labels show degeneracy/tie saturation in the completed
  one-cell diagnostic; a zero-terminal-value problem and a learned historical
  tail such as `Q_H + V_tail_hat` remain possible future explanations/designs.
  The historical-tail diagnostic must precede any actual new loss definition.
  Wulver owns the current historical-tail readiness work, and this local audit
  did not contact or synchronize Wulver.
- decision-rule branch:
  hard argmin may be unreliable when predicted candidate values are close
  relative to uncertainty. Future candidates include margin-gated softmax,
  uncertainty/Thompson-style selection, confidence-gated LRU fallback, and
  selective learned override of LRU. These are conditional future experiments,
  not current primary method changes.
- learning branch:
  low fractions `1%`, `2%`, `5%`, and `10%` have a validated local checkpoint;
  the `25%` phase completed naturally and locally on `2026-08-09` (verified
  `2026-08-10`); `50%` first attempt was `INTERRUPTED_BEFORE_FIRST_COMPLETED_UNIT`
  (launched `2026-08-10` outside tmux, killed when that SSH session
  closed, `0/42` rows); relaunched same day in tmux session
  `kbs_learning_curve_50pct_20260810`, now `RUNNING_LOCAL_RESUME`; `100%`
  remains TODO.
- continuation branch:
  C1 is `Q_H^LRU -> pi1`; C2 is `Q_H^pi1 -> pi2`. The local implementation is
  ready, but Wulver real-data validation and later source/result sync are still
  required before any result claim.
