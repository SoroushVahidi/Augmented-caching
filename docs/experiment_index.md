# Experiment index

This index maps major experiment families to their runners, configurations, and results.

| Experiment | Role | Runner | Config | Results (Tracked) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **KBS Canonical** | CANONICAL | `scripts/experiments/canonical/run_policy_comparison_wulver_v1.py` | (Implicit in script/Slurm) | `analysis/manuscript_canonical/` | FROZEN |
| **Fairness Sweep** | PRIMARY_REVIEWER | `scripts/experiments/reviewer/run_family_winner_selection.py` | `configs/reviewer/reviewer_fairness_protocol.json` | `analysis/reviewer_revision/` | ACTIVE |
| **LRB Baseline** | SENSITIVITY | `scripts/experiments/reviewer/run_lrb_external_baseline.py` | (CLI params) | `analysis/external_learned_baselines/lrb/` | RUNNING |
| **Objective Ablation**| PRIMARY_REVIEWER | `scripts/experiments/reviewer/run_objective_ablation.py` | `configs/reviewer/supervision_objective_ablation_v1.json`| `analysis/reviewer_revision/` | ACTIVE |
| **Guard Ablation** | DIAGNOSTIC | `scripts/experiments/reviewer/run_fallback_guard_ablation.py` | (CLI params) | `analysis/exploratory/` | COMPLETE |
| **Joint State** | SENSITIVITY | `scripts/experiments/exploratory/run_joint_state_reasoning_ablation.py` | (CLI params) | `analysis/exploratory/` | ARCHIVAL |

## Experiment Status Definitions

- **ACTIVE**: Currently being executed or analyzed for the reviewer revision.
- **FROZEN**: Canonical results intended for the final manuscript submission.
- **RUNNING**: Long-running job currently active on a compute node.
- **COMPLETE**: Data collection finished; results available.
- **ARCHIVAL**: Historical experiment kept for reference; not part of current revision.
- **DIAGNOSTIC**: Short-term audit or sanity check.
