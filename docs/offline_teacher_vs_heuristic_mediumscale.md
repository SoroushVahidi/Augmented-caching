# Offline-Teacher vs Heuristic Supervision (Medium-Scale Controlled Evaluation)

This run is a stronger follow-up to the lightweight comparison in `docs/offline_teacher_vs_heuristic_experiment.md`.

## Goal

Test whether offline-teacher supervision creates a **real downstream replay advantage** once we scale local evaluation while still keeping controlled conditions.

## Controlled setup

The experiment keeps these fixed across the two supervision sources:

- same features (`EVICT_VALUE_V1_FEATURE_COLUMNS` + capacity + horizon),
- same model family per run (`ridge` by default; optional `random_forest`),
- same trace pool,
- same capacities and horizon,
- same seed-based train/val/test assignment.

Only the supervision source changes:

- `heuristic`: rollout-based labels,
- `offline_teacher`: exact Belady (when applicable) or LP-approx teacher labels.

## What is scaled up vs the tiny run

- Uses more capacities by default (`2,3,4,5` instead of only a tiny pair).
- Uses repeated seeded splits (`0..5` by default) to increase held-out diversity.
- Aggregates downstream replay across many seed × trace × capacity workloads.
- Adds systematic wins/ties/losses and workload ranking summaries.
- Adds disagreement analysis by both family and capacity.

## Command

```bash
python scripts/run_offline_teacher_vs_heuristic_mediumscale.py \
  --trace-glob data/example_unweighted.json,data/example_atlas_v1.json,data/example_general_caching.json,data/example.json \
  --capacities 2,3,4,5 \
  --horizon 16 \
  --seeds 0,1,2,3,4,5 \
  --output-dir analysis/offline_teacher_vs_heuristic_mediumscale
```

Optional second model family:

```bash
python scripts/run_offline_teacher_vs_heuristic_mediumscale.py \
  --model-family random_forest \
  --output-dir analysis/offline_teacher_vs_heuristic_mediumscale_rf
```

## Outputs

- `results.csv`: regression + decision metrics by seed/split/source.
- `downstream_results.csv`: per workload replay misses/hit-rate + delta vs LRU.
- `disagreement_by_family.csv`: disagreement rate by trace family.
- `disagreement_by_capacity.csv`: disagreement rate by capacity.
- `workload_rankings.csv`: rank by misses within each workload.
- `model_comparison.csv`: pairwise heuristic vs offline_teacher wins/ties/losses.
- `gain_by_decision.csv`: teacher-vs-heuristic regret gain per decision.
- `summary.json`: machine-readable aggregate summary.
- `report.md`: concise interpretation and recommendation.

## Decision criteria

Prefer `offline_teacher` when all (or most) are true:

1. lower total misses than heuristic over aggregate held-out workloads,
2. at least neutral delta-vs-LRU compared with heuristic,
3. wins > losses in `model_comparison.csv`,
4. gains are concentrated in disagreement regions (`gain_by_decision.csv`).

If mixed, carry both labels forward and expand workload diversity next.
