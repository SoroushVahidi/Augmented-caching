# Learning-Augmented Weighted Paging (`lafc`)

Implementation of Baseline 1 from:

> Bansal, Coester, Kumar, Purohit, Vee.
> **"Learning-Augmented Weighted Paging"**.
> SODA 2022.

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
- Objective: minimise total fetch cost.

---

## Input Format (JSON)

```json
{
  "requests":    ["A", "B", "C", "A", "B", "D", "A", "C", "B", "A"],
  "weights":     {"A": 1.0, "B": 2.0, "C": 4.0, "D": 1.0},
  "predictions": [3, 4, 7, 6, 8, 9, 10, 9999, 9999, 9999]
}
```

`predictions` is optional; if omitted, the runner generates perfect
predictions from the trace automatically.

---

## Available Policies

| CLI name             | Description                                        |
|----------------------|----------------------------------------------------|
| `lru`                | Least-Recently-Used                                |
| `weighted_lru`       | Evicts the cheapest (min-weight) cached page       |
| `advice_trusting`    | Evicts page with max predicted next-arrival        |
| `la_det`             | Deterministic LA weighted paging (paper Theorem 1) |

---

## Example Commands

### Smoke test

```bash
python -m lafc.runner.run_policy \
    --policy la_det \
    --trace  data/example.json \
    --capacity 3
```

### Compare all baselines

```bash
for policy in lru weighted_lru advice_trusting la_det; do
    python -m lafc.runner.run_policy \
        --policy   $policy \
        --trace    data/example.json \
        --capacity 3
done
```

---

## Output Files (written to `--output-dir`, default `output/`)

| File                    | Contents                                         |
|-------------------------|--------------------------------------------------|
| `summary.json`          | policy, total_cost, hits, misses, hit_rate       |
| `metrics.json`          | prediction error η, per-class weighted surprises |
| `per_step_decisions.csv`| one row per request: t, page, hit, cost, evicted |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## See Also

`docs/baselines.md` — detailed algorithm description, interpretation notes,
and paper-to-code mapping.
