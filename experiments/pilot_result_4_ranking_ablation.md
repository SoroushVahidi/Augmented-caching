# Pilot Result 4: Ranking-Criteria Ablation

## Scope and honesty note

This repository currently does **not** expose a ready GSM8K + API-backed harness in the checked-in code path, so this run uses a lightweight synthetic offline pilot to isolate ranking behavior under matched budgets.

## Exact commands used

```bash
python -m experiments.run_pilot_ranking_ablation \
  --samples 40 \
  --budget 8 \
  --branch-count 5 \
  --min-expand 2 \
  --seed 7 \
  --run-id 20260412_ranking_ablation
```

## Provider/model

- provider: `synthetic_offline`
- model: `none`

## Pilot size

- 40 instances per method

## Budget settings (matched)

- action budget: 8
- min initial expansions per branch: 2
- branch count: 5

## Ranking formulas used

- `raw_score`: `rank_value = score`
- `score_plus_progress`: `rank_value = score + 0.10 * depth`
- `relative_rank`: `rank_value = rank(score) + rank(depth)` using active-pool rank positions

## Metrics table

| Method | accuracy | avg actions | avg expansions | avg verifications | avg surviving branches | budget exhaustion rate |
|---|---:|---:|---:|---:|---:|---:|
| greedy_single_path | 0.225 | 8.0 | 8.0 | 1.0 | 5.000 | 0.000 |
| best_of_n | 0.150 | 8.0 | 40.0 | 1.0 | 4.350 | 0.000 |
| fixed_width_beam | 0.150 | 8.0 | 24.0 | 1.0 | 5.000 | 0.000 |
| adaptive_min_expand + raw_score | 0.125 | 8.0 | 8.0 | 1.0 | 4.825 | 0.000 |
| adaptive_min_expand + score_plus_progress | 0.300 | 8.0 | 8.0 | 1.0 | 4.825 | 0.000 |
| adaptive_min_expand + relative_rank | 0.150 | 8.0 | 8.0 | 1.0 | 4.725 | 0.000 |

## Short comparison

- Best ranking rule in this pilot: **`score_plus_progress`**.
- Did relative ranking help? **Not in this run** (roughly tied with beam/best-of-n and above raw_score, but below score+progress).
- Did adding progress help? **Yes in this run** (`score_plus_progress` materially exceeded `raw_score` under equal budgets).
- Stability/noise: with only 40 samples, results are **directional and noisy**, not conclusive. A larger run is needed before claiming robust gains.

## Output directory

- `outputs/pilot/20260412_ranking_ablation/`
