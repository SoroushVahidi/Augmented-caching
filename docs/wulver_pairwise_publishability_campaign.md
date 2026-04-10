# Wulver pairwise publishability campaign

This campaign tests whether pairwise-ranking supervision can support a publishable empirical story even if theorem guarantees are weak.

Primary questions:

1. Does pairwise supervision improve offline eviction-decision quality vs best-candidate regret?
2. Does that translate to online policy gains against strong baselines?
3. Is behavior robust under noisy labels and cross-family splits?

## Added scripts

- `scripts/run_pairwise_publishability_campaign_task.py`
  - per-task dataset generation (from rollout candidates), pairwise model training, offline+online eval
  - model families: logistic, random forest, hist gradient boosting
  - label variants: `head_pair`, `regret_diff`
  - robustness: train-label corruption `label_noise` in `{0.0,0.1,0.2}`
- `scripts/aggregate_pairwise_publishability_campaign.py`
  - aggregates all task outputs
  - writes campaign summary + report + W/T/L tables + offline ablation summary

## Baselines included (online)

- `lru`
- `blind_oracle`
- `predictive_marker`
- `trust_and_doubt`
- `robust_ftp_d_marker`
- `ml_gate_v2`
- `evict_value_v1`
- `evict_value_v1_guarded`
- `atlas_v3`
- `rest_v1`
- plus learned `evict_value_pairwise_v1` (artifact model from each task)

## Slurm files

- `slurm/pairwise_publishability_campaign_smoke.sbatch`
- `slurm/pairwise_publishability_campaign_array.sbatch`
- `slurm/pairwise_publishability_campaign_aggregate.sbatch`

## Launch sequence

```bash
SMOKE_JOB_ID=$(sbatch --parsable slurm/pairwise_publishability_campaign_smoke.sbatch)
ARRAY_JOB_ID=$(sbatch --parsable --dependency=afterok:${SMOKE_JOB_ID} --array=0-53 --export=ALL,EXP_TAG=pairwise_pub_r1,TRACE_CHUNK=3,MAX_REQ=40000,CAPACITIES=32,64,128,HORIZONS=4,8,16 slurm/pairwise_publishability_campaign_array.sbatch)
AGG_JOB_ID=$(sbatch --parsable --dependency=afterok:${ARRAY_JOB_ID} slurm/pairwise_publishability_campaign_aggregate.sbatch)
```

## Output locations

Per-task:

- `analysis/pairwise_publishability_campaign/jobs/<job_label>/summary.json`
- `analysis/pairwise_publishability_campaign/jobs/<job_label>/offline_metrics.csv`
- `analysis/pairwise_publishability_campaign/jobs/<job_label>/offline_per_family.csv`
- `analysis/pairwise_publishability_campaign/jobs/<job_label>/offline_per_horizon.csv`
- `analysis/pairwise_publishability_campaign/jobs/<job_label>/online_metrics.csv`
- `analysis/pairwise_publishability_campaign/jobs/<job_label>/wtl_vs_baselines.csv`

Aggregated:

- `analysis/pairwise_publishability_campaign/offline_metrics_all.csv`
- `analysis/pairwise_publishability_campaign/online_metrics_all.csv`
- `analysis/pairwise_publishability_campaign/wtl_vs_baselines_all.csv`
- `analysis/pairwise_publishability_campaign/offline_ablation_summary.csv`
- `analysis/pairwise_publishability_campaign/wtl_collapsed.csv`
- `analysis/pairwise_publishability_campaign/campaign_summary.json`
- `analysis/pairwise_publishability_campaign/report.md`

## Interpretation contract

Final recommendation in the report is one of:

- `promising enough for manuscript mainline`
- `promising only as appendix/ablation`
- `not promising enough`
