# KBS second-revision artifact map

This document maps each reviewer concern to its code, configuration, generated
outputs, current status, and evidence caveats.

Status labels used here:

- `COMPLETE_VALIDATED`
- `PARTIAL`
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
| 28-model registry freeze | `scripts/build_supervision_objective_ablation_registry.py`, `scripts/experiments/supervision_objective_ablation_gates.py` | preserved local registry in `/home/soroush/Augmented-caching-objective-ablation/analysis/supervision_objective_ablation_v1/model_registry.json` | `COMPLETE_VALIDATED` | Usable provenance evidence | Registry copy is currently preserved in the sibling objective-ablation worktree |
| Same-example audit | `scripts/experiments/audit_supervision_objective_examples.py` | sibling objective-ablation worktree `.../same_example_audit.json` | `COMPLETE_VALIDATED` | Usable provenance evidence | Final audit exists locally but has not been copied into the canonical worktree's untracked `analysis/` tree |
| Fairness audit | `scripts/experiments/audit_supervision_objective_fairness.py` | sibling objective-ablation worktree `.../fairness_audit.json` | `COMPLETE_VALIDATED` | Usable provenance evidence | Same caveat as above |

Reverified aggregate misses from `analysis/supervision_objective_ablation_v1/policy_comparison.csv`:

- `objective_pairwise`: `565127`
- `objective_reuse_distance`: `571456`
- `objective_next_arrival`: `573059`
- `objective_eviction_loss`: `601569`

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
