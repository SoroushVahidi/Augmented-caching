# Slurm batch templates

Cluster job definitions for Wulver-scale (and other) runs. **Paths and `sbatch` examples in `docs/wulver_*.md` and batch headers are authoritative** if this README lags.

## Canonical KBS Wulver `heavy_r1`

| File | Stage |
|------|--------|
| [`evict_value_v1_wulver_heavy_train.sbatch`](evict_value_v1_wulver_heavy_train.sbatch) | Dataset build + summary + train |
| [`evict_value_v1_wulver_heavy_eval.sbatch`](evict_value_v1_wulver_heavy_eval.sbatch) | Full manifest policy comparison (replay) |
| [`evict_value_v1_wulver_heavy_smoke.sbatch`](evict_value_v1_wulver_heavy_smoke.sbatch) | Wiring check only — **not** canonical KBS numbers |

**Runbook:** [`../docs/wulver_heavy_evict_value_experiment.md`](../docs/wulver_heavy_evict_value_experiment.md)  
**Checklist:** [`../CANONICAL_KBS_SUBMISSION.md`](../CANONICAL_KBS_SUBMISSION.md)

## Exploratory / non-canonical drivers (examples)

- `evict_value_v1_wulver_multi_phase.sbatch`, dataset-only drivers — see comments in each file and `scripts/README.md`.

Logs default under `slurm/logs/` per batch files.
