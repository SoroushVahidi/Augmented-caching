# KBS second-revision artifact map

This document maps each reviewer concern to its code, configuration, generated
outputs, current status, and evidence caveats.

Status labels used here:

- `COMPLETE_VALIDATED`
- `RUNNING_LOCAL`
- `PARTIAL`
- `DIAGNOSTIC_PARTIAL`
- `SMOKE_ONLY`
- `PENDING_CONTROLLED_RUN`
- `CONTAMINATED_DO_NOT_USE`
- `HISTORICAL`
- `BLOCKED_PENDING_SYNC`

## Source-of-truth notes

- Canonical code branch: `kbs/second-revision-science`
- Current local repository-state note:
  [`../kbs_second_revision_repository_state.md`](../kbs_second_revision_repository_state.md)
- Wulver-only orchestration files are still missing locally as of 2026-08-09
  because non-interactive SSH inspection failed on authentication. Treat that
  sync as `BLOCKED_PENDING_SYNC`; do not fabricate those files.

## Reviewer #2 Major 1: learned-baseline comparison

| Item | Code / protocol | Current local output | Status | Eligibility | Caveats |
|---|---|---|---|---|---|
| LRB | `scripts/experiments/run_reviewer_fairness.py`, `docs/reviewer_fairness_protocol.md` | `analysis/reviewer_fairness/policy_comparison_lrb.csv` | `COMPLETE_VALIDATED` | Primary reviewer table: use only `policy_variant=primary_controlled_window` rows | `deployment_full_stream` rows are supporting context only |
| 3L-Cache | same fairness protocol | `analysis/reviewer_fairness/policy_comparison_three_l_cache.csv` | `COMPLETE_VALIDATED` | Primary reviewer table: controlled-window rows only | Batch-size sensitivity belongs to supporting analysis, not the primary table |
| HALP | same fairness protocol | `analysis/reviewer_fairness/policy_comparison_halp.csv` | `COMPLETE_VALIDATED` | Primary reviewer table: controlled-window rows only | Training-resource parity with `evict_value_v1` is not claimed |
| CACHEUS | same fairness protocol | `analysis/reviewer_fairness/policy_comparison_cacheus.csv` | `COMPLETE_VALIDATED` | Primary reviewer table: controlled-window rows only | Official source preserved; upstream fixed seed remains part of provenance |
| Non-learned baselines (`lru`, `sieve`, `fifo_reinsertion`, `blind_oracle_lru_combiner`, `rest_v1`, `trust_and_doubt`, `predictive_marker`, `offline_belady`) | same fairness protocol | `analysis/reviewer_fairness/policy_comparison_*.csv` | `COMPLETE_VALIDATED` | Controlled-window rows are usable for the primary table | `offline_belady` is oracle context, not a deployable baseline |
| Held-out `evict_value_v1` retraining | `scripts/experiments/run_evict_cross_family_pipeline.py`, `scripts/experiments/run_evict_cross_family_heldout_eval.py`, `docs/reviewer_fairness_cross_family_v1.md` | `analysis/reviewer_fairness_cross_family_v1/` plus preserved local data/models under the fairness worktree | `PARTIAL` | Not yet usable for the primary table until the held-out evaluation rows exist | `scripts/train_evict_value_wulver_v1.py` memory-bounded trainer changes are already semantically integrated in canonical commit `710b854` |
| Old fair-window `evict_value_v1` comparison | `analysis/reviewer_fairness/policy_comparison_evict_value_v1.csv` | same file | `CONTAMINATED_DO_NOT_USE` | Never use in the primary reviewer table | Training/test overlap is explicitly recorded in the file itself and in the overlap audit |

## Reviewer #2 Major 2: supervision-objective ablation

| Item | Code / protocol | Current local output | Status | Eligibility | Caveats |
|---|---|---|---|---|---|
| Objective sweep (`eviction_loss`, `reuse_distance`, `next_arrival`, `pairwise`) | `docs/supervision_objective_ablation_protocol.md`, `scripts/experiments/run_supervision_objective_ablation_eval.py` | `analysis/supervision_objective_ablation_v1/policy_comparison.csv` | `COMPLETE_VALIDATED` | Usable supporting evidence | Aggregate misses must be taken from the frozen CSV, not from stale notes |
| 28-model registry freeze | `scripts/build_supervision_objective_ablation_registry.py`, `scripts/experiments/supervision_objective_ablation_gates.py` | preserved local registry in sibling objective-ablation worktree `../Augmented-caching-objective-ablation/analysis/supervision_objective_ablation_v1/model_registry.json` | `COMPLETE_VALIDATED` | Usable provenance evidence | Registry copy is currently preserved in the sibling objective-ablation worktree |
| Same-example audit | `scripts/experiments/audit_supervision_objective_examples.py` | sibling objective-ablation worktree `.../same_example_audit.json` | `COMPLETE_VALIDATED` | Usable provenance evidence | Final audit exists locally but has not been copied into the canonical worktree's untracked `analysis/` tree |
| Fairness audit | `scripts/experiments/audit_supervision_objective_fairness.py` | sibling objective-ablation worktree `.../fairness_audit.json` | `COMPLETE_VALIDATED` | Usable provenance evidence | Same caveat as above |

Reverified aggregate misses from `analysis/supervision_objective_ablation_v1/policy_comparison.csv`:

- `objective_pairwise`: `565127`
- `objective_reuse_distance`: `571456`
- `objective_next_arrival`: `573059`
- `objective_eviction_loss`: `601569`

Important naming distinction:

- `objective_pairwise` in this frozen ablation is a different supervision
  objective from `objective_eviction_loss`;
- `eviction_loss_pairwise` in the newer learning-curve diagnostic is not that
  earlier condition. It derives pairwise labels from the same underlying
  eviction-loss scalar labels and therefore isolates representation rather than
  target construction.

## Supplementary local diagnostic: same-target scalar-vs-pairwise learning curve

| Item | Code / protocol | Current local output | Status | Eligibility | Caveats |
|---|---|---|---|---|---|
| Same-target scalar-vs-pairwise learning curve | `configs/supervision_objective_learning_curve_v1.json`, `scripts/experiments/run_supervision_objective_learning_curve.py`, `src/lafc/reviewer_diagnostics.py`, `tests/test_supervision_objective_learning_curve.py` | `analysis/supervision_objective_learning_curve_v1/`, `models/supervision_objective_learning_curve_v1/` | `RUNNING_LOCAL` | `DIAGNOSTIC_PARTIAL` while active or partially complete | Explanatory diagnostic only, not a primary reviewer comparison |

Diagnostic intent:

- compare `eviction_loss_scalar` against `eviction_loss_pairwise`,
- hold the underlying eviction-loss notion fixed,
- vary the amount of training data by fraction,
- test whether pairwise representation alone appears more sample-efficient than
  scalar regression.

Current locally audited checkpoint to preserve:

- validated low-fraction cells:
  fractions `1%, 2%, 5%, 10%`
- completed families at that audited checkpoint:
  `brightkite, citibike, cloudphysics, metacdn`
- validated completed units:
  `16`
- validated result rows:
  `96`

Current local high-fraction extension status:

- active tmux session:
  `kbs_learning_curve_highfrac_20260809`
- currently launched fraction phase:
  `25%`
- target scope for the active phase:
  all seven held-out folds
- local wall-time budget:
  `10` hours
- later planned phases:
  `50%`, then `100%`
- current status:
  `RUNNING_LOCAL`
- final conclusion status:
  none yet; do not infer scientific results from partial `25%` outputs until a
  completed unit is audited

Same-example fairness guarantee:

- one deterministic nested decision ordering per fold,
- each fraction reuses an exact prefix of that ordering,
- scalar rows use every candidate row belonging to the selected decision ids,
- pairwise rows are derived only from those exact filtered scalar rows,
- the runner is fail-closed on nested-subset violations, mismatched pairwise
  label source, missing decision ids, and tie-retention errors.

Resume and isolation notes:

- rows are keyed by `(condition, fraction, held_out_family, capacity)` and
  written incrementally to an isolated directory,
- unit completion is checkpointed at the `(held_out_family, fraction)` level in
  `campaign_state.json`,
- completed cells can be audited independently,
- partial campaign state must be preserved as-is at clean wall-time stop.
- the validated `1%` to `10%` checkpoint and the running `25%` extension must
  remain analytically separate until the active phase reaches a clean stop.

## Reviewer #2 Major 3 and Reviewer #3: distribution-shift diagnosis

| Item | Code / protocol | Current local output | Status | Eligibility | Caveats |
|---|---|---|---|---|---|
| Local resumed distribution-shift campaign | `docs/distribution_shift_ablation_protocol.md`, `scripts/experiments/run_distribution_shift_ablation.py` | `analysis/distribution_shift_ablation_v1/` | `PARTIAL` | Diagnostic-only; not a final seven-family result | Current local checkpoint: `24/42` primary rows, `4/7` families complete |
| Trajectory-divergence diagnostics | same protocol | `analysis/distribution_shift_ablation_v1/trajectory_divergence.csv` | `PARTIAL` | Diagnostic-only | Divergence alone is not causal evidence |
| State-shift diagnostics | same protocol | `analysis/distribution_shift_ablation_v1/state_shift_metrics.csv` | `PARTIAL` | Diagnostic-only | Reduced measured shift did not automatically improve misses in the local `metacdn` result |
| Isolated-family Wulver continuation | Wulver-only runner and Slurm scripts not yet synced back | not present locally | `BLOCKED_PENDING_SYNC` | Not citable from this branch until the source/orchestration is synced and inspected | Missing local files: `run_distribution_shift_family.py` and the related `slurm/kbs_distribution_shift_wulver*.sbatch` drivers |

## Reviewer #2 Major 4: practical significance

| Item | Code / protocol | Current local output | Status | Eligibility | Caveats |
|---|---|---|---|---|---|
| Exact-decision-preserving optimization smoke | `docs/practical_significance_ablation_protocol.md`, `scripts/experiments/run_practical_significance_ablation.py` | `analysis/practical_significance_ablation_v1/exact_optimization_equivalence.json` | `SMOKE_ONLY` | Supporting implementation evidence only | `all_variants_exact_across_all_trace_capacity_pairs=true`, but `speedup_numbers_are_final_reviewer_evidence=false` |
| Break-even and miss-cost sweep | same protocol | `analysis/practical_significance_ablation_v1/break_even_miss_cost.csv`, `miss_cost_sweep.csv` | `SMOKE_ONLY` | Supporting analysis only | Uses smoke-scale timing, not controlled final timing |
| Controlled timing campaign | same protocol | no final controlled result present locally | `PENDING_CONTROLLED_RUN` | Not yet usable | Must remain separate from smoke conclusions |

## Historical heavy-run material

| Item | Current role |
|---|---|
| `analysis/*_heavy_r1.*` and `scripts/paper/build_kbs_main_manuscript_artifacts.py` inputs | `HISTORICAL` builder/provenance line |
| `docs/wulver_heavy_evict_value_experiment.md` | `HISTORICAL` runbook |
| `docs/evict_value_v1_kbs_canonical_artifacts.md` | `HISTORICAL` filename map |

## Standard cache-metric coverage audit

This branch already captures some standard cache metrics directly and can
derive others cheaply from frozen CSVs. The table below is a repository
organization note, not a manuscript claim.

| Metric | Classification | Notes |
|---|---|---|
| Miss ratio | `AVAILABLE_FROM_EXISTING_ARTIFACTS` | Present in frozen reviewer CSVs and diagnostic CSVs |
| Runtime / throughput proxy | `AVAILABLE_FROM_EXISTING_ARTIFACTS` | `runtime_seconds` is present in frozen reviewer CSVs; controlled final timing is still separate from smoke timing |
| Worst-family regret / robustness | `CHEAP_DERIVATION` | Can be derived from current per-family controlled-window CSV rows without retraining |
| Capacity-wise relative behavior (`32/64/128`) | `CHEAP_DERIVATION` | Present in the frozen reviewer CSVs and objective-ablation CSVs |
| Variance across workloads | `CHEAP_DERIVATION` | Per-family rows already exist; summary statistics are a post-processing step |
| Decision-quality metrics (`validation_top1`, `validation_mean_regret`, pairwise accuracy) | `AVAILABLE_FROM_EXISTING_ARTIFACTS` for `analysis/supervision_objective_learning_curve_v1/`; otherwise `REQUIRES_NEW_REPLAY` | The learning-curve diagnostic writes them directly, but the primary reviewer fairness CSVs do not |
| Top-k candidate agreement | `REQUIRES_NEW_REPLAY` | Current frozen reviewer outputs do not retain candidate-set rankings |
| Byte miss ratio / bytes fetched | `REQUIRES_NEW_REPLAY` | Some trace families have size metadata, but the current frozen reviewer CSVs record only unit object-slot semantics |
| Weighted miss cost using real trace costs | `REQUIRES_NEW_DATA` | Current reviewer-science traces and outputs do not preserve heterogeneous real miss costs for the frozen comparisons |
| Runtime memory footprint | `REQUIRES_NEW_REPLAY` | Model files exist, but controlled runtime RSS profiling was not recorded in the frozen reviewer CSVs |
| Workload-locality / churn stratification | `CHEAP_DERIVATION` | Requires adding trace-stat summaries and joining them with existing per-family result rows; no retraining needed |
