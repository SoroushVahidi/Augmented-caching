# heavy_smoke execution audit

**Date (UTC):** 2026-04-12  
**Repository root:** `/project/ikoutis/sv96/Augmented-caching`

## Submission

Submitted **only** the stock smoke batch (no `sbatch` overrides to partition, QoS, CPUs, memory, walltime, or `EXP_TAG`). Defaults match `slurm/evict_value_v1_wulver_heavy_smoke.sbatch`:

- `partition=general`, `qos=standard`, `nodes=1`, `ntasks=1`, `cpus-per-task=8`, `mem=32G`, `time=02:00:00`, CPU-only (no GPU).

Command:

```bash
cd /project/ikoutis/sv96/Augmented-caching
sbatch --parsable slurm/evict_value_v1_wulver_heavy_smoke.sbatch
```

**Job ID:** `910353` (`evictv1-heavy-smoke`)

## Slurm completion

```bash
sacct -j 910353 -n -o JobID,State,ExitCode,Elapsed -P
```

| Field | Value |
|--------|--------|
| State | `COMPLETED` |
| ExitCode | `0:0` |
| Elapsed | `00:02:45` |

Logs:

- `slurm/logs/evictv1-heavy-smoke-910353.out`
- `slurm/logs/evictv1-heavy-smoke-910353.err` (empty on success)

## Output verification

```bash
for f in \
  analysis/evict_value_wulver_v1_train_metrics_heavy_smoke.json \
  analysis/evict_value_wulver_v1_model_comparison_heavy_smoke.csv \
  analysis/evict_value_wulver_v1_best_config_heavy_smoke.json \
  analysis/evict_value_wulver_v1_policy_comparison_heavy_smoke.csv \
  analysis/evict_value_wulver_v1_policy_comparison_heavy_smoke.md; do
  test -s "$f" && wc -c "$f" || echo "MISSING_OR_EMPTY $f"
done
```

| File | Bytes (approx.) | Status |
|------|-----------------|--------|
| `analysis/evict_value_wulver_v1_train_metrics_heavy_smoke.json` | 5375 | non-empty |
| `analysis/evict_value_wulver_v1_model_comparison_heavy_smoke.csv` | 1096 | non-empty |
| `analysis/evict_value_wulver_v1_best_config_heavy_smoke.json` | 184 | non-empty |
| `analysis/evict_value_wulver_v1_policy_comparison_heavy_smoke.csv` | 2707 | non-empty |
| `analysis/evict_value_wulver_v1_policy_comparison_heavy_smoke.md` | 1228 | non-empty |

## KBS manuscript scope (from docs)

Per `docs/kbs_manuscript_workflow.md` and `docs/kbs_manuscript_submission_index.md`, **`heavy_smoke` is a wiring check**, not the canonical KBS Wulver line. Main quantitative claims require `EXP_TAG=heavy_r1` and `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`.

## Verdict

**Smoke job succeeded:** exit `0:0`, all five expected `*_heavy_smoke` artifacts present and non-empty. Safe to proceed toward heavier canonical jobs when cluster resources allow; do **not** substitute smoke outputs for `heavy_r1` manuscript tables.
