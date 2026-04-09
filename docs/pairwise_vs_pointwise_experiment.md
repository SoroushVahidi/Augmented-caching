# Pairwise vs Pointwise Supervision Experiment (Evict Value)

## Motivation

Eviction is inherently a **ranking** decision: at a miss with a full cache, we need to pick *which candidate is better to evict than the others*.

Pointwise regression predicts an absolute score/regret per candidate. Pairwise supervision instead trains on direct comparisons `(candidate_i better than candidate_j)` and may better match the downstream decision structure.

## Controlled comparison

This experiment keeps fixed:

- same traces,
- same capacities,
- same horizon,
- same feature pipeline (`EVICT_VALUE_V1_FEATURE_COLUMNS` + capacity/horizon context),
- same seed-based train/val/test split logic,
- same replay-style downstream simulator.

Only supervision/objective style changes:

- **Pointwise**: Ridge regression on candidate-level regret labels.
- **Pairwise**: Logistic regression on pairwise delta features; decision-time ranking reconstructed via pairwise vote aggregation.

## Method summary

1. Build candidate rows with heuristic rollout labels.
2. Build pairwise rows from the same candidate decisions.
3. Train pointwise and pairwise models on each seed split.
4. Evaluate candidate-choice quality on train/val/test.
5. Replay learned policies on held-out traces and capacities.
6. Aggregate wins/ties/losses, rank, and regime breakdowns.

## Commands

```bash
python scripts/run_pairwise_vs_pointwise_experiment.py \
  --trace-glob data/example_unweighted.json,data/example_atlas_v1.json,data/example_general_caching.json,data/example.json \
  --capacities 2,3,4,5 \
  --horizon 16 \
  --seeds 0,1,2,3,4,5 \
  --output-dir analysis/pairwise_vs_pointwise
```

Optional style-specific runs:

```bash
python scripts/run_pairwise_vs_pointwise_experiment.py --supervision-style pointwise
python scripts/run_pairwise_vs_pointwise_experiment.py --supervision-style pairwise
```

## Outputs

- `results.csv`: split-level candidate-choice metrics by supervision style.
- `downstream_results.csv`: replay misses/hit-rate and delta vs LRU.
- `disagreement_analysis.csv`: hard/tied/close/disagreement/family/capacity analyses.
- `summary.json`: aggregate downstream and wins/ties/losses.
- `report.md`: concise interpretation.

## Limitations

- Local medium-scale traces only (not Wulver-scale diversity).
- Pairwise vote aggregation is a simple tournament-style heuristic.
- Pairwise may still need stronger calibration and richer trace families.
