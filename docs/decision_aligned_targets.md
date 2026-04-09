# Decision-aligned target improvement package (evict_value)

## Motivation (conservative framing)

The current candidate-scoring direction is promising, but purely pointwise or myopic labels can be weakly aligned with the real online decision:

> On each miss with a full cache, choose **which candidate page to evict**.

This package keeps the same policy structure (score candidates, evict best-scored) and improves only the supervision targets to be closer to the decision objective.

This is **not** full RL, and **not** a repo redesign.

---

## Targets implemented

For each eviction decision and each candidate victim `v`, with finite horizon `H`:

1. **Raw short-horizon rollout loss**

   `loss_H(v)` = number of unit-cost misses over next `H` requests
   after forcing eviction of `v` now and inserting current request page,
   then continuing with a configured continuation policy.

2. **Regret-aligned target**

   `regret_H(v) = loss_H(v) - min_u loss_H(u)`

   So the best candidate(s) at each decision have regret `0`.

3. **Pairwise decision target**

   For each candidate pair `(i, j)` from the same decision:

   - `label_i_better = 1` iff `regret_H(i) < regret_H(j)`
   - include `rollout_regret_diff = regret_H(i) - regret_H(j)`

   Ties can be skipped (default) or included (`--include-ties`).

4. **Multi-horizon support**

   Builders accept multiple horizons, e.g. `--horizons 4,8,16,32`.

5. **Optional discounted per-row targets**

   If `--discount-gamma > 0`, each candidate row also includes:

   - `rollout_loss_discounted = rollout_loss_h * gamma^(h-1)`
   - `rollout_regret_discounted = rollout_regret_h * gamma^(h-1)`

---

## Continuation policy used in rollouts

Configurable via `--continuation-policy`:

- `lru` (default; recommended first-check baseline)
- `blind_oracle`

This is a practical finite-horizon approximation, not infinite-horizon cost-to-go.

---

## Scripts

### 1) Build decision-aligned candidate dataset

```bash
python scripts/build_evict_value_decision_aligned_dataset.py \
  --trace-glob "data/example_*.json" \
  --capacities 2,3,4 \
  --horizons 4,8,16,32 \
  --continuation-policy lru \
  --max-rows 250000 \
  --max-requests-per-trace 0 \
  --output-dir data/derived/evict_value_decision_aligned \
  --sample-seed 7
```

Outputs (text-only):

- `candidate_rows.csv`
- `family_summary.csv`
- `dataset_summary.json`

### 2) Build pairwise dataset

```bash
python scripts/build_evict_value_pairwise_dataset.py \
  --candidate-csv data/derived/evict_value_decision_aligned/candidate_rows.csv \
  --output-dir data/derived/evict_value_pairwise \
  --max-rows 500000 \
  --sample-seed 7
```

Outputs:

- `pairwise_rows.csv`
- `dataset_summary.json`

### 3) Decision-aligned first-check experiment

```bash
python scripts/run_evict_value_decision_aligned_first_check.py \
  --candidate-csv data/derived/evict_value_decision_aligned/candidate_rows.csv \
  --output-dir analysis/evict_value_decision_aligned_first_check
```

Reported metrics include:

- MAE / RMSE for rollout loss
- MAE / RMSE for rollout regret
- top-1 candidate accuracy
- mean chosen regret
- mean regret vs best
- per-family and per-horizon CSVs

### 4) Pairwise first-check experiment

```bash
python scripts/run_evict_value_pairwise_first_check.py \
  --pairwise-csv data/derived/evict_value_pairwise/pairwise_rows.csv \
  --output-dir analysis/evict_value_pairwise_first_check
```

Reported metrics include:

- pairwise accuracy
- top-1 candidate accuracy (reconstructed from pairwise probabilities)
- mean chosen regret
- mean regret vs best
- per-family and per-horizon CSVs

---

## Assumptions and simplifications

- Miss cost is unit-cost in label simulation.
- Rollout horizon is finite and user-configurable.
- Continuation policy is fixed per dataset build.
- This package does not claim to solve long-horizon control; it is a practical test bed for target alignment.

---

## Extension path toward `evict_value_v2`

Low-risk next steps:

1. Compare online policy quality when training on:
   - legacy pointwise target,
   - rollout regret target,
   - pairwise target.
2. Add model-selection by validation **decision metrics** (not only MAE).
3. Optionally blend horizons and continuation policies to reduce label brittleness.

