# heavy_r1 readiness (canonical manuscript evidence)

**Readiness:** `NOT_READY_CLUSTER_BLOCKED`

**Timestamp (UTC):** 2026-04-12

## Summary

Slurm job **910352** (`evictv1-heavy-eval`, canonical `EXP_TAG=heavy_r1`) remains **PENDING** with **`(ReqNodeNotAvail, Reserved for maintenance)`**. It has **not** started; runtime is zero; **no** canonical policy comparison artifacts have been written.

## Files

| Path | Status |
|------|--------|
| `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` | **Missing** |
| `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.md` | **Missing** |
| `models/evict_value_wulver_v1_best_heavy_r1.pkl` | Present (prerequisite; not re-verified byte-by-byte this session) |
| `data/derived/evict_value_v1_wulver_heavy_r1/manifest.json` | Present (prerequisite) |
| `slurm/logs/evictv1-heavy-eval-910352.out` / `.err` | **Not created yet** (job never allocated) |

## Manuscript bundle

**Not rebuilt from full canonical policy evidence** in this session — policy CSV absent. After **910352** completes with exit **0:0**, run from repo root:

```bash
export PYTHONPATH="${PYTHONPATH:-$(pwd)/src}"
test -s analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv
python scripts/paper/build_kbs_main_manuscript_artifacts.py
```

Expect `"policy_comparison_present": true` in `reports/manuscript_artifacts/manuscript_artifact_manifest.json` and regenerated `figure2_*`, `figure3_*`, Table~3 from `heavy_r1` CSV only.

## If the cluster unblocks later

1. Monitor **910352** with `squeue` / `sacct` until `COMPLETED` and `ExitCode=0:0`.
2. If **910352** is **FAILED**, **CANCELLED**, or **TIMEOUT** before producing CSV/MD: diagnose `slurm/logs/evictv1-heavy-eval-910352.*` (once present), then resubmit **only** `slurm/evict_value_v1_wulver_heavy_eval.sbatch` with `EXP_TAG=heavy_r1` — **do not** rerun heavy training unless artifacts are proven corrupt.

## Readiness enum

This note uses exactly: **`NOT_READY_CLUSTER_BLOCKED`**
