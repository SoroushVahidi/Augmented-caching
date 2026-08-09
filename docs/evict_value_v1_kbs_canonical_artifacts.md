# Historical `heavy_r1` artifact set and current second-revision context

This file records the exact `*_heavy_r1` filenames used by the older Wulver
manuscript-support builder path. It remains useful for provenance and for the
historical builder in `scripts/paper/`, but it is not the main orientation
document for the current KBS second-revision reviewer-science branch.

For current reviewer-science evidence, start at:

- [`kbs_manuscript_workflow.md`](kbs_manuscript_workflow.md)
- [`reviewer/kbs_second_revision_artifact_map.md`](reviewer/kbs_second_revision_artifact_map.md)
- [`reviewer/kbs_evidence_eligibility.md`](reviewer/kbs_evidence_eligibility.md)

## Historical `heavy_r1` pipeline

1. [`wulver_heavy_evict_value_experiment.md`](wulver_heavy_evict_value_experiment.md)
2. `slurm/evict_value_v1_wulver_heavy_train.sbatch`
3. `slurm/evict_value_v1_wulver_heavy_eval.sbatch`

## Historical builder inputs

These are the exact files consumed by the older KBS builder path:

| Role | Path |
|------|------|
| Policy comparison (CSV) | `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` |
| Policy comparison (report) | `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.md` |
| Dataset summary | `analysis/evict_value_v1_wulver_dataset_summary_heavy_r1.md` |
| Training metrics | `analysis/evict_value_wulver_v1_train_metrics_heavy_r1.json` |
| Model comparison | `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv` |
| Best config | `analysis/evict_value_wulver_v1_best_config_heavy_r1.json` |
| Trace manifest | `analysis/wulver_trace_manifest_full.csv` |

## Why this file still exists

The repository still keeps the `heavy_r1` filenames because:

- older manuscript-support tooling refers to them explicitly,
- they remain part of the historical provenance of the project,
- they help distinguish older heavy-run inputs from newer reviewer-science
  evidence directories.

## What this file does not mean

This document does not imply that the historical `heavy_r1` line is the only
or current canonical path for the branch `kbs/second-revision-science`. The
current reviewer-science workflow is broader and includes:

- learned-baseline fairness comparisons,
- supervision-objective ablation,
- distribution-shift diagnosis,
- practical-significance analysis.

Those are mapped in
[`reviewer/kbs_second_revision_artifact_map.md`](reviewer/kbs_second_revision_artifact_map.md).
