# Learning-Augmented Caching (`lafc`)

Implementations of learning-augmented caching algorithms from:

**Baseline 1:**
> Bansal, Coester, Kumar, Purohit, Vee.
> **"Learning-Augmented Weighted Paging"**.
> SODA 2022.

**Baseline 3:**
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

- Cache holds at most **k** pages (unit sizes).
- Each page `p` has a fetch cost (weight) `w_p > 0`.
- Requests arrive one at a time: `σ_1, σ_2, ..., σ_T`.
- At time `t`, the algorithm also receives a prediction `τ_t` for the **next
  arrival time** of `σ_t`.
- If `σ_t` is not in cache, the algorithm pays `w_{σ_t}` and fetches the page.
- Objective: minimise total fetch cost (= number of misses for unit weights).

---

## Input Format (JSON)

```json
{
  "requests":    ["A", "B", "C", "A", "B", "D", "A", "C", "B", "A"],
  "weights":     {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0},
  "predictions": [3, 4, 7, 6, 8, 9, 10, 9999, 9999, 9999]
}
```

`predictions` is optional; if omitted, the runner generates perfect
predictions from the trace automatically.

Two example traces are provided:

| File | Description |
|------|-------------|
| `data/example_unweighted.json` | Unit-weight trace (for TRUST&DOUBT / Baseline 3) |
| `data/example.json`            | Weighted trace with heterogeneous costs (for Baseline 1) |

---

## Available Policies

| CLI name               | Description                                           | Paper           |
|------------------------|-------------------------------------------------------|-----------------|
| `lru`                  | Least-Recently-Used (classical baseline)              | —               |
| `weighted_lru`         | Evicts cheapest (min-weight) cached page              | —               |
| `marker`               | Deterministic LRU-Marker (O(log k)-robust)            | Fiat et al. '91 |
| `blind_oracle`         | Blind Oracle / FTP: evicts page with max predicted_next | ICML 2020     |
| `follow_the_prediction`| Follow-The-Prediction (alias of blind_oracle)         | ICML 2020       |
| `predictive_marker`    | Prediction-guided Marker eviction                     | Lykouris '18    |
| `trust_and_doubt`      | **TRUST&DOUBT** (main Baseline 3)                     | ICML 2020       |
| `advice_trusting`      | Advice-trusting Belady (weighted version)             | —               |
| `la_det`               | Deterministic LA weighted paging (Baseline 1)         | SODA 2022       |

---

## Example Commands

### Smoke test — TRUST&DOUBT

```bash
python -m lafc.runner.run_policy \
    --policy trust_and_doubt \
    --trace  data/example_unweighted.json \
    --capacity 3
```

### Toy experiment — compare all Baseline 3 policies

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
