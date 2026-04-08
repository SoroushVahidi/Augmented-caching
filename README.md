# Learning-Augmented Caching (`lafc`)

This repository contains three learning-augmented caching baselines:

## Baseline 1 — Learning-Augmented Weighted Paging

Implementation of:

> Bansal, Coester, Kumar, Purohit, Vee.
> **"Learning-Augmented Weighted Paging"**.
> SODA 2022.

## Baseline 2 — Predictive Marker (Competitive Caching with Machine Learned Advice)

Implementation of:

> Lykouris, Vassilvitskii.
> **"Competitive Caching with Machine Learned Advice"**.
> ICML 2018 / JACM 2021.

## Baseline 3 — TRUST&DOUBT (Online Metric Algorithms with Untrusted Predictions)

Implementation of:

> Antoniadis, Coester, Eliáš, Polak, Simon.
> **"Online Metric Algorithms with Untrusted Predictions"**.
> ICML 2020.  arXiv:2003.03033.

---

## Installation

```bash
pip install -e ".[dev]"
```

---

## Problem Setting

### Baseline 1 (Weighted Paging)

- Cache holds at most **k** pages (unit sizes).
- Each page `p` has a fetch cost (weight) `w_p > 0`.
- Requests arrive one at a time: `σ_1, σ_2, ..., σ_T`.
- At time `t`, the algorithm also receives a prediction `τ_t` for the **next
  arrival time** of `σ_t`.
- If `σ_t` is not in cache, the algorithm pays `w_{σ_t}` and fetches the page.
- Objective: minimise total fetch cost.

### Baselines 2 and 3 (Unweighted / Standard Paging)

- Cache holds at most **k** pages (unit sizes).
- All misses cost **1** (unit cost).
- Requests arrive one at a time with optional predictions.
- Objective: minimise total number of cache misses.

---

## Input Format (JSON)

### Baseline 1 (weighted paging)

```json
{
  "requests":    ["A", "B", "C", "A", "B", "D", "A", "C", "B", "A"],
  "weights":     {"A": 1.0, "B": 2.0, "C": 4.0, "D": 1.0},
  "predictions": [3, 4, 7, 6, 8, 9, 10, 9999, 9999, 9999]
}
```

### Baselines 2 and 3 (unweighted paging)

```json
{
  "requests":    ["A", "B", "C", "A", "B", "D", "A", "C", "B", "A"],
  "predictions": [3, 4, 7, 6, 8, 9, 10, 9999, 9999, 9999]
}
```

`predictions` is optional; if omitted, the runner generates perfect
predictions from the trace automatically.  `weights` is optional for
Baselines 2 and 3 (unit weights are used by Marker / BlindOracle /
PredictiveMarker / TrustAndDoubt regardless).

Two example traces are provided:

| File | Description |
|------|-------------|
| `data/example_unweighted.json` | Unit-weight trace (Baselines 2 and 3) |
| `data/example.json`            | Weighted trace with heterogeneous costs (Baseline 1) |

---

## Available Policies

| CLI name               | Description                                           | Paper            |
|------------------------|-------------------------------------------------------|------------------|
| `lru`                  | Least-Recently-Used (classical baseline)              | —                |
| `weighted_lru`         | Evicts cheapest (min-weight) cached page              | —                |
| `advice_trusting`      | Advice-trusting Belady (weighted version)             | —                |
| `la_det`               | Deterministic LA weighted paging (Baseline 1)         | SODA 2022        |
| `marker`               | Deterministic LRU-Marker (O(log k)-robust)            | Fiat et al. '91  |
| `blind_oracle`         | Blind Oracle / FTP: evicts page with max predicted_next | ICML 2018/2020 |
| `follow_the_prediction`| Follow-The-Prediction (alias of blind_oracle)         | ICML 2020        |
| `predictive_marker`    | Prediction-guided Marker eviction (Baseline 2)        | ICML 2018        |
| `trust_and_doubt`      | **TRUST&DOUBT** (Baseline 3)                          | ICML 2020        |

---

## Example Commands

### Smoke test — TRUST&DOUBT (Baseline 3)

```bash
python -m lafc.runner.run_policy \
    --policy trust_and_doubt \
    --trace  data/example_unweighted.json \
    --capacity 3
```

### Toy experiment — compare all Baseline 2/3 policies

```bash
for policy in lru marker blind_oracle predictive_marker trust_and_doubt; do
    echo "=== $policy ==="; \
    python -m lafc.runner.run_policy \
        --policy   $policy \
        --trace    data/example_unweighted.json \
        --capacity 3; \
done
```

### With perfect predictions

```bash
python -m lafc.runner.run_policy \
    --policy trust_and_doubt \
    --trace  data/example_unweighted.json \
    --capacity 3 \
    --perfect-predictions
```

### Smoke test — Baseline 1 (weighted paging)

```bash
python -m lafc.runner.run_policy \
    --policy la_det \
    --trace  data/example.json \
    --capacity 3
```

### Compare all Baseline 1 policies

```bash
for policy in lru weighted_lru advice_trusting la_det; do
    echo "=== $policy ==="; \
    python -m lafc.runner.run_policy \
        --policy   $policy \
        --trace    data/example.json \
        --capacity 3; \
done
```

---

## Output Files (written to `--output-dir`, default `output/`)

| File                     | Contents                                         |
|--------------------------|--------------------------------------------------|
| `summary.json`           | policy, total_cost, hits, misses, hit_rate       |
| `metrics.json`           | prediction error η, per-class weighted surprises |
| `per_step_decisions.csv` | one row per request: t, page, hit, cost, evicted |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## See Also

`docs/baselines.md` — detailed algorithm descriptions, interpretation notes,
and paper-to-code mappings for all implemented baselines.
