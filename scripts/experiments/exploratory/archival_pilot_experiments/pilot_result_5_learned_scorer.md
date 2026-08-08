# Pilot Result 5: First Learned Branch Scorer (v1)

## Commands used

```bash
python -m experiments.run_pilot_ranking_ablation \
  --samples 60 \
  --budget 8 \
  --branch-count 5 \
  --min-expand 2 \
  --seed 11 \
  --run-id 20260413_branch_trace_seed

python scripts/train_branch_scorer.py \
  --trace-input outputs/pilot/20260413_branch_trace_seed \
  --out-dir outputs/models/branch_scorer_lr_v1

python -m experiments.run_pilot_ranking_ablation \
  --samples 60 \
  --budget 8 \
  --branch-count 5 \
  --min-expand 2 \
  --seed 19 \
  --compare-learned-only \
  --learned-model-path outputs/models/branch_scorer_lr_v1 \
  --run-id 20260413_learned_branch_scorer
```

## Training target used

- `eventually_correct_branch`
- Definition: `1` iff `branch_id == gold_branch_id` for that episode; else `0`.
- Limitation: proxy for branch promise, not exact continuation-value label.

## Feature list

- `raw_score`
- `depth`
- `expansions`
- `verifications`
- `remaining_budget`
- `post_verify_score`
- `score_delta`
- `survived_pruning_steps`

## Model type

- Logistic regression (`sklearn LogisticRegression(class_weight='balanced', max_iter=1000)`)

## Provider/model for branch generation

- provider: `synthetic_offline`
- model: `none`

## Metric table (pilot comparison)

| Method | accuracy | avg actions | avg expansions | avg verifications | avg surviving branches | budget exhaustion rate |
|---|---:|---:|---:|---:|---:|---:|
| fixed_width_beam | 0.233 | 8.0 | 24.0 | 8.517 | 4.983 | 0.000 |
| adaptive_score_plus_progress | 0.233 | 8.0 | 8.0 | 2.850 | 4.733 | 0.000 |
| adaptive_relative_rank | 0.300 | 8.0 | 8.0 | 2.467 | 4.650 | 0.000 |
| adaptive_learned_branch_score | 0.200 | 8.0 | 8.0 | 2.867 | 4.733 | 0.000 |

## Honest interpretation

- Did learned scorer help? **Not in this pilot**. It underperformed both adaptive hand-designed rankers.
- Better than hand-designed ranking? **No**. `adaptive_relative_rank` was best here.
- Is target too weak/noisy? **Likely yes**. The v1 proxy label is coarse for local ranking updates.
- Training signal quality: training AUC was only about `0.53`, which is close to weakly-informative.

## Recommended next improvements

1. Build cleaner continuation labels (counterfactual utility of one extra expansion) instead of only final-branch identity.
2. Add train/validation split by episode and report out-of-sample AUC.
3. Keep model simple (still linear) but add per-feature standardization + coefficient inspection.
4. If label quality improves, retest learned scorer against `relative_rank` with larger pilots.
