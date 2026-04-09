# Offline-Teacher vs Heuristic Supervision Experiment

This experiment answers:

> Does offline-theory-guided supervision improve learned eviction scoring vs older heuristic supervision?

## What is compared

- `heuristic`: rollout-based decision-aligned labels.
- `offline_teacher`: exact Belady labels when valid, LP-approx teacher labels otherwise.

Both sides use:

- same feature pipeline (`EVICT_VALUE_V1_FEATURE_COLUMNS` + capacity/horizon),
- same model family (Ridge + StandardScaler),
- same trace split logic (trace-hash train/val/test),
- same capacities/horizon settings.

Only label source changes.

## Command

```bash
python scripts/run_offline_teacher_vs_heuristic_experiment.py \
  --trace-glob data/example_*.json,data/example_general_caching.json \
  --capacities 2,3 \
  --horizon 12 \
  --output-dir analysis/offline_teacher_vs_heuristic
```

## Outputs

- `analysis/offline_teacher_vs_heuristic/results.csv`
- `analysis/offline_teacher_vs_heuristic/downstream_results.csv`
- `analysis/offline_teacher_vs_heuristic/disagreement_by_family.csv`
- `analysis/offline_teacher_vs_heuristic/summary.json`
- `analysis/offline_teacher_vs_heuristic/report.md`

## Metrics included

- train/val/test row counts by label source
- MAE/RMSE on target regret
- top-1 candidate accuracy
- mean chosen regret
- mean regret vs best candidate
- disagreement rate (`teacher_best != heuristic_best`)
- disagreement concentration by family
- lightweight downstream replay misses/hit-rate vs LRU on held-out traces

## Notes

- This is intentionally lightweight and local; no large sweeps.
- Results should be interpreted with disagreement concentration: gains mainly on disagreement regions indicate new supervision signal from offline teachers.
