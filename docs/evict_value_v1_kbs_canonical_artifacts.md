# Canonical `evict_value_v1` artifacts for KBS (heavy_r1 only)

**Navigation:** `docs/kbs_manuscript_workflow.md` (full manuscript path: Slurm → analysis inputs → `build_kbs_main_manuscript_artifacts.py` → `tables/manuscript/`, `figures/manuscript/`).

This document fixes a single source of truth for the main Wulver evaluation and training evidence used with `scripts/paper/build_kbs_main_manuscript_artifacts.py`.

**See also:** `docs/kbs_manuscript_submission_index.md` (reviewer-facing index), `analysis/README.md` (canonical vs non-canonical root filenames).

## Canonical pipeline

1. `docs/wulver_heavy_evict_value_experiment.md`
2. `slurm/evict_value_v1_wulver_heavy_train.sbatch` → `EXP_TAG=heavy_r1`
3. `slurm/evict_value_v1_wulver_heavy_eval.sbatch` → `EXP_TAG=heavy_r1`

## Canonical analysis files (committed inputs to the manuscript builder)

| Role | Path |
|------|------|
| Policy comparison (CSV) | `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` |
| Policy comparison (report) | `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.md` |
| Dataset summary (human-readable) | `analysis/evict_value_v1_wulver_dataset_summary_heavy_r1.md` |
| Training metrics | `analysis/evict_value_wulver_v1_train_metrics_heavy_r1.json` |
| Model comparison (selection / ablation) | `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv` |
| Best config | `analysis/evict_value_wulver_v1_best_config_heavy_r1.json` |
| Trace manifest | `analysis/wulver_trace_manifest_full.csv` |

The manuscript builder reads exactly these paths (see `EVIDENCE_FILES` in `scripts/paper/build_kbs_main_manuscript_artifacts.py`).

## Generated manuscript outputs (from the builder)

- `tables/manuscript/table{1-4}_*.{csv,tex}`
- `figures/manuscript/figure{1-4}_*.{pdf,png}`
- `reports/manuscript_artifacts/{manuscript_artifact_manifest.json,manuscript_artifact_report.md,latex_snippets/*.tex}`

## Unsuffixed files: not canonical for KBS main results

The following are **legacy or alternate-driver** outputs and must **not** be treated as the heavy_r1 main comparison unless proven identical:

- `analysis/evict_value_wulver_v1_policy_comparison.csv` / `.md` — default filename from `run_policy_comparison_wulver_v1.py`; in-repo content has included policies outside the heavy eval baseline set (e.g. `atlas_v3`, `ml_gate_v1`, `ml_gate_v2`).
- `analysis/evict_value_wulver_v1_train_metrics.json`, `..._model_comparison.csv`, `..._best_config.json` — parallel to older or multi-phase runs; canonical training artifacts use the `_heavy_r1` suffix above.
- `analysis/evict_value_v1_wulver_dataset_summary.md` — unsuffixed summary; canonical is `..._dataset_summary_heavy_r1.md`.

## Missing heavy_r1 policy comparison

If `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` is absent, run the heavy eval stage (after successful training) per `docs/wulver_heavy_evict_value_experiment.md`. The manuscript builder will fail with a missing-file error until this file exists.
