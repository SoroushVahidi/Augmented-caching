# KBS Local To Wulver Sync Status

Date: 2026-08-09

Scope: local repository audit only. No Wulver contact was performed during this
pass, so Wulver runtime state is intentionally not asserted here.

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
  - active/running local diagnostic output
  - currently includes 25% units for `brightkite`, `citibike`,
    `cloudphysics`, and `metacdn`; the phase is not complete
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
