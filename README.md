# Learning-Augmented Caching (`lafc`)

This repository contains learning-augmented caching baselines:

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
> ICML 2020.

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

### Baseline 2 (Unweighted / Standard Paging)

- Cache holds at most **k** pages (unit sizes).
- All misses cost **1** (unit cost).
- Requests arrive one at a time with optional predictions.
- Objective: minimise total number of cache misses.

---

## Input Format (JSON or CSV)

### Baseline 1 (weighted paging)

```json
{
  "requests":    ["A", "B", "C", "A", "B", "D", "A", "C", "B", "A"],
  "weights":     {"A": 1.0, "B": 2.0, "C": 4.0, "D": 1.0},
  "predictions": [3, 4, 7, 6, 8, 9, 10, 9999, 9999, 9999]
}
```

### Baseline 2 (unweighted paging)

```json
{
  "requests":    ["A", "B", "C", "A", "B", "D", "A", "C", "B", "A"],
  "predictions": [3, 4, 7, 6, 8, 9, 10, 9999, 9999, 9999]
}
```

`predictions` is optional; if omitted, default is `inf` predictions. `weights` is optional for
unweighted baselines (unit weights are used).


For TRUST&DOUBT, provide either:
- `predicted_caches` in JSON (list of cache states aligned with requests), or
- `--derive-predicted-caches` to convert next-arrival predictions to state predictions.

CSV format requires `page_id` and optional `predicted_next`, `predicted_cache` (pipe-separated pages).

---

## Available Policies

| CLI name             | Baseline | Description                                        |
|----------------------|----------|----------------------------------------------------|
| `lru`                | 1        | Least-Recently-Used                                |
| `weighted_lru`       | 1        | Evicts the cheapest (min-weight) cached page       |
| `advice_trusting`    | 1        | Evicts page with max predicted next-arrival        |
| `la_det`             | 1        | Deterministic LA weighted paging (paper Theorem 1) |
| `marker`             | 2        | Standard Marker (phase-based, unit cost)           |
| `blind_oracle`       | 2        | Blind Oracle: evict argmax predicted next-arrival  |
| `predictive_marker`  | 2        | Predictive Marker (Lykouris & Vassilvitskii 2018)  |
| `trust_and_doubt`   | 3        | TRUST&DOUBT (Antoniadis et al. 2020)               |

---

## Example Commands

### Smoke test (Baseline 1)

```bash
python -m lafc.runner.run_policy \
    --policy la_det \
    --trace  data/example.json \
    --capacity 3
```

### Smoke test (Baseline 2 — Predictive Marker)

```bash
python -m lafc.runner.run_policy \
    --policy predictive_marker \
    --trace  data/example_unweighted.json \
    --capacity 3
```

### Compare all Baseline 1 policies

```bash
for policy in lru weighted_lru advice_trusting la_det; do
    python -m lafc.runner.run_policy \
        --policy   $policy \
        --trace    data/example.json \
        --capacity 3
done
```

### Compare all Baseline 2 policies

```bash
for policy in marker blind_oracle predictive_marker; do
    echo "=== $policy ===" && \
    python -m lafc.runner.run_policy \
        --policy   $policy \
        --trace    data/example_unweighted.json \
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


---

## Dataset Support

Supported benchmark families:

- BrightKite check-ins
- CitiBike NYC trip traces
- SPEC CPU2006 memory traces (manual local ingestion)
- wiki2018 CDN trace (manual local ingestion by default)

Prepare datasets via:

```bash
python scripts/datasets/prepare_all.py --dataset <brightkite|citibike|spec_cpu2006|wiki2018|all>
```

Processed traces are written under `data/processed/<dataset>/`.
See `docs/datasets.md` for source links, legal caveats, mapping assumptions, and exact commands.
