# Wulver campaign: pairwise-ranking chain-witness theorem validation

This campaign is a compute-heavy search for violations/support of three candidate lemmas:

1. Chain monotonicity: `a(e1) < a(e2) < ...` along stale-victim chains.
2. Head-local witness bound: `D(c) <= |I(c)| + 1`.
3. Phase/global budget bound: `sum_c |I(c)| <= eta_pair`.

The campaign runs as unattended Slurm jobs and writes machine-readable artifacts plus a final markdown report.

## Added scripts

- `scripts/run_pairwise_chain_witness_theorem_checks.py`
  - per-job theorem checker + counterexample minimizer
  - workload modes: `exhaustive`, `random`, `structured`, `real`
- `scripts/aggregate_pairwise_chain_witness_campaign.py`
  - merges per-job outputs and builds campaign-level summary/report

## Added Slurm files

- `slurm/pairwise_chain_witness_campaign_smoke.sbatch`
  - smoke wiring test (`random`, small instance count)
- `slurm/pairwise_chain_witness_campaign_array.sbatch`
  - high-compute array campaign (mode-sharded tasks)
- `slurm/pairwise_chain_witness_campaign_aggregate.sbatch`
  - final aggregation/report step

## Launch sequence

## 1) Smoke run first

```bash
sbatch --export=ALL,EXP_TAG=campaign_smoke slurm/pairwise_chain_witness_campaign_smoke.sbatch
```

## 2) Main high-compute campaign (array)

```bash
ARRAY_JOB_ID=$(sbatch --parsable --array=0-23 --export=ALL,EXP_TAG=campaign_r1,MAX_INSTANCES=25000,MAX_LEN=11,REAL_PREFIX=30000 slurm/pairwise_chain_witness_campaign_array.sbatch)
```

## 3) Aggregation after array completes

```bash
sbatch --dependency=afterok:${ARRAY_JOB_ID} slurm/pairwise_chain_witness_campaign_aggregate.sbatch
```

## Outputs

Per-job outputs:

- `analysis/pairwise_chain_witness_campaign/jobs/<job_label>/summary.json`
- `analysis/pairwise_chain_witness_campaign/jobs/<job_label>/instances.csv`
- `analysis/pairwise_chain_witness_campaign/jobs/<job_label>/chains.csv`
- `analysis/pairwise_chain_witness_campaign/jobs/<job_label>/violations.csv`
- `analysis/pairwise_chain_witness_campaign/jobs/<job_label>/minimized_counterexamples.csv`

Aggregated outputs:

- `analysis/pairwise_chain_witness_campaign/all_instances.csv`
- `analysis/pairwise_chain_witness_campaign/all_violations.csv`
- `analysis/pairwise_chain_witness_campaign/all_minimized_counterexamples.csv`
- `analysis/pairwise_chain_witness_campaign/job_summaries.json`
- `analysis/pairwise_chain_witness_campaign/campaign_summary.json`
- `analysis/pairwise_chain_witness_campaign/report.md`

Slurm logs:

- `slurm/logs/pairwise-chain-smoke-<jobid>.out/.err`
- `slurm/logs/pairwise-chain-camp-<arrayjobid>_<taskid>.out/.err`
- `slurm/logs/pairwise-chain-agg-<jobid>.out/.err`

## Monitoring

```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
```

## Notes

- CPU-only campaign by design.
- The checker is intentionally conservative and does not alter canonical policy code paths.
- Real-trace mode uses `analysis/wulver_trace_manifest_full.csv` prefixes for compute efficiency.
