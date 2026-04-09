# Decision-aligned eviction targets (experimental v2 plan)

## Why this next step?

`evict_value_v1` moved the project toward direct candidate scoring, but its supervision is still mostly pointwise and short-horizon.
That can be myopic for eviction because the online action is a **relative choice** among candidates at each miss.

This v2 package is an offline, text-only experimental upgrade that keeps the same interface:

- score each candidate,
- evict the lowest-harm candidate.

The change is in supervision: rollout cost-to-go labels and ranking-aware pairwise labels.

## What is implemented

### 1) Rollout / cost-to-go candidate labels

For each eviction decision, each candidate victim, and each horizon `H`:

1. Force-evict the candidate.
2. Insert current request page.
3. Simulate the next `H` requests under a configurable continuation policy.
4. Count unit-cost misses over that horizon.

This gives:

- `rollout_loss_h(candidate)`
- `rollout_regret_h(candidate) = rollout_loss_h(candidate) - min_candidate rollout_loss_h`

Supported continuation policies in v2 dataset builder:

- `lru` (default)
- `blind_oracle` (Belady-style victim choice over the finite rollout window)

### 2) Pairwise ranking labels

From each decision's candidate set, generate pair rows `(i, j)`:

- `label_i_better = 1` if `rollout_regret_h(i) < rollout_regret_h(j)`
- else `0` (ties can be skipped by default)

This directly supervises relative candidate quality for the actual ranking decision.

## Relation to cited decision-learning references (conceptual)

- **Predict-then-optimize / Smart Predict-then-Optimize** motivates that proxy prediction quality and decision quality can diverge; we therefore log decision-level regret metrics in addition to MAE.
- **Decision-focused ranking perspective** supports using ranking-aware surrogates when the deployed action is an argmin/argmax over alternatives.
- **DAgger / no-regret IL / AGGREVATE** motivate cost-to-go style supervision for action selection under sequential covariate shift; here we implement an offline finite-horizon approximation rather than full interactive learning.

This is intentionally practical and conservative, not full end-to-end decision-focused optimization and not full RL.

## Why this is practical now

- Reuses existing `evict_value_v1` candidate generation and feature logic.
- Uses finite-horizon rollout labels that are easy to compute offline.
- Produces only text artifacts (`csv`, `json`, `md`), no model checkpoints.
- Enables hypothesis testing quickly before heavier online integration.

## Caveats

- Finite-horizon rollout is still an approximation to long-run cost-to-go.
- Label quality depends on continuation policy choice (`lru` vs `blind_oracle`).
- Pairwise ranking remains a surrogate objective for final miss count.

## Scripts and commands

Build rollout candidate dataset:

```bash
python scripts/build_evict_value_v2_rollout_dataset.py \
  --trace-glob "data/example_*.json" \
  --dataset examples \
  --capacities 2,3,4 \
  --horizons 4,8,16,32 \
  --reference-policy lru \
  --output-dir data/derived/evict_value_v2_rollout
```

Build pairwise dataset from candidate rows:

```bash
python scripts/build_evict_value_v2_pairwise_dataset.py \
  --candidate-csv data/derived/evict_value_v2_rollout/candidate_rows.csv \
  --output-dir data/derived/evict_value_v2_pairwise
```

Run rollout first-check experiment:

```bash
python scripts/run_evict_value_v2_rollout_first_check.py \
  --candidate-csv data/derived/evict_value_v2_rollout/candidate_rows.csv \
  --output-dir analysis/evict_value_v2_rollout_first_check
```

Run pairwise first-check experiment:

```bash
python scripts/run_evict_value_v2_pairwise_first_check.py \
  --pairwise-csv data/derived/evict_value_v2_pairwise/pairwise_rows.csv \
  --output-dir analysis/evict_value_v2_pairwise_first_check
```
