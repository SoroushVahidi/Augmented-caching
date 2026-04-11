# Wulver Slurm job report — Augmented-caching (`sv96`)

**Cluster:** `wulver` (`ClusterName=wulver`, queried 2026-04-11).  
**Scope:** Jobs with `WorkDir` under `/mmfs1/project/ikoutis/sv96/Augmented-caching` (`sacct -u sv96 -S 2025-01-01`, pipe-separated).

## Summary counts (89 accounting rows: main job IDs only)

| State     | Count |
|-----------|-------|
| COMPLETED | 54    |
| FAILED    | 20    |
| TIMEOUT   | 15    |

## Cancelled pending aggregates (user `scancel`)

These depended on `afterok` for full arrays that had non-success tasks; they stayed **PD** with `DependencyNeverSatisfied` until cancelled.

| JobID | Name                 | Final state |
|-------|----------------------|-------------|
| 909441 | `pairwise-chain-agg` | CANCELLED   |
| 909467 | `pairwise-pub-agg`   | CANCELLED   |

## Notable singleton jobs

| JobID  | Name                 | State   | Notes                                      |
|--------|----------------------|---------|--------------------------------------------|
| 908325 | `evictv1-heavy-train`| COMPLETED | KBS heavy training stage (~18.5 h)      |
| 908326 | `evictv1-heavy-eval` | TIMEOUT | 24 h walltime; eval did not finish cleanly |
| 907377 | `evictv1-multi`      | COMPLETED | Multi-phase pipeline (~24 h)            |
| 907022–907035 | `evictv1-dsgen` | mixed   | Early dataset-gen attempts               |
| 909439 | `pairwise-chain-smoke` | COMPLETED | |
| 909465 | `pairwise-pub-smoke`   | COMPLETED | |

## Array 909440 — `pairwise-chain-camp` (tasks 0–23)

- **COMPLETED:** 22  
- **TIMEOUT:** 2 (`909440_22`, `909440_23`, 16 h limit)

## Array 909466 — `pairwise-publish` (tasks 0–53)

- **COMPLETED:** 24  
- **FAILED:** 18 (`909466_18` … `909466_35`, exit 1:0)  
- **TIMEOUT:** 12 (20 h limit; tasks include `909466_1`, `_4`, `_7`, `_10`, `_13`, `_16`, `_37`, `_40`, `_43`, `_46`, `_49`, `_52`)

Campaign aggregation (`pairwise_publishability_campaign_aggregate.sbatch`) was not run successfully under the original `afterok` chain.

## This commit (artifact sync)

Adds **11** per-task output directories under `jobs/` that were present on disk but not yet tracked in git (same layout as other `pairwise_pub_r1_*` tasks: `summary.json`, `offline_metrics.csv`, `online_metrics.csv`, `wtl_vs_baselines.csv`, etc.).

Reproduce listing of project-scoped jobs:

```bash
sacct -u sv96 -S 2025-01-01 \
  --format=JobID,JobName,State,ExitCode,Elapsed,Timelimit,Start,End,Partition,WorkDir -P -n \
  | grep '/mmfs1/project/ikoutis/sv96/Augmented-caching'
```

See also: `docs/wulver_pairwise_publishability_campaign.md`, `slurm/pairwise_publishability_campaign_*.sbatch`.
