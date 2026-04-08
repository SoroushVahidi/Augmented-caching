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

## Experimental Framework — `atlas_v1` / `atlas_v2` / `atlas_v3` / `atlas_cga_v1` / `atlas_cga_v2` / `rest_v1`

This repository also includes **experimental** framework policies, `atlas_v1`,
`atlas_v2`, `atlas_v3`, `atlas_cga_v1`, `atlas_cga_v2`, and `rest_v1`, for unweighted paging with bucketed predictions and optional
confidence scores. `atlas_v2` adds dynamic trust adaptation over `atlas_v1`, while
`atlas_v3` introduces confidence-calibrated local trust by prediction context.
`atlas_cga_v1` is a calibration-guided extension that uses online safe-to-evict calibration
probabilities to scale predictor influence.
`atlas_cga_v2` is a hierarchical/context-sharing refinement of CGA that shares calibration
signal across global, bucket, confidence-bin, and full-context levels.
`rest_v1` is an abstention/selective-trust pivot: instead of confidence blending,
it gates between TRUST (predictor eviction) and ABSTAIN (LRU) using per-context
online regret-style trust updates.
Both are intended for empirical study only (no theorem guarantee is claimed).

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

For `atlas_v1` / `atlas_v2` / `atlas_v3` / `atlas_cga_v1` / `atlas_cga_v2` / `rest_v1`, the preferred optional JSON extension is:

```json
{
  "requests": ["A", "B", "C"],
  "prediction_records": [
    {"bucket": 0, "confidence": 0.9},
    {"bucket": 2, "confidence": 0.4},
    {"bucket": 3}
  ]
}
```

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
| `atlas_v1`          | Exp      | Experimental confidence-aware policy with LRU fallback |
| `atlas_v2`          | Exp      | Experimental confidence-aware policy with dynamic trust adaptation |
| `atlas_v3`          | Exp      | Experimental confidence-aware local-trust policy (CCLT v1) |
| `atlas_cga_v1`      | Exp      | Experimental calibration-guided local-trust policy (CGA v1) |
| `atlas_cga_v2`      | Exp      | Experimental hierarchical context-sharing calibration policy (CGA v2) |
| `rest_v1`           | Exp      | Experimental ReST selective-trust/abstention gating policy |

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

### Smoke test (`atlas_v1`, experimental)

```bash
python -m lafc.runner.run_policy \
    --policy atlas_v1 \
    --trace data/example_atlas_v1.json \
    --capacity 3 \
    --default-confidence 0.5 \
    --bucket-source trace
```

### Smoke test (`atlas_v2`, experimental)

```bash
python -m lafc.runner.run_policy \
    --policy atlas_v2 \
    --trace data/example_atlas_v1.json \
    --capacity 3 \
    --default-confidence 0.5 \
    --bucket-source trace \
    --atlas-window 32 \
    --atlas-rho 0.3 \
    --atlas-initial-gamma 0.8 \
    --atlas-mismatch-threshold 2
```

### Smoke test (`atlas_v3`, experimental)

```bash
python -m lafc.runner.run_policy \
    --policy atlas_v3 \
    --trace data/example_atlas_v1.json \
    --capacity 3 \
    --default-confidence 0.5 \
    --bucket-source trace \
    --atlas-initial-local-trust 0.7 \
    --atlas-confidence-bins 0.33,0.66 \
    --atlas-eta-pos 0.03 \
    --atlas-eta-neg 0.12 \
    --atlas-bucket-regret-mode linear \
    --atlas-tie-epsilon 1e-9
```

### Smoke test (`atlas_cga_v1`, experimental)

```bash
python -m lafc.runner.run_policy \
    --policy atlas_cga_v1 \
    --trace data/example_atlas_v1.json \
    --capacity 3 \
    --default-confidence 0.5 \
    --bucket-source trace \
    --atlas-initial-local-trust 0.7 \
    --atlas-confidence-bins 0.33,0.66 \
    --atlas-calibration-prior-a 1.0 \
    --atlas-calibration-prior-b 1.0 \
    --atlas-calibration-min-support 5 \
    --atlas-calibration-shrinkage 10 \
    --atlas-safe-horizon-mode bucket_regret
```

### Smoke test (`atlas_cga_v2`, experimental)

```bash
python -m lafc.runner.run_policy \
    --policy atlas_cga_v2 \
    --trace data/example_atlas_v1.json \
    --capacity 3 \
    --default-confidence 0.5 \
    --bucket-source trace \
    --atlas-hier-global-prior-a 1.0 \
    --atlas-hier-global-prior-b 1.0 \
    --atlas-hier-min-support 5 \
    --atlas-hier-weight-mode normalized_support \
    --atlas-hier-shrink-strength 10
```

### Smoke test (`rest_v1`, experimental selective trust)

```bash
python -m lafc.runner.run_policy \
    --policy rest_v1 \
    --trace data/example_atlas_v1.json \
    --capacity 3 \
    --default-confidence 0.5 \
    --bucket-source trace \
    --rest-initial-trust 0.5 \
    --rest-eta-pos 0.05 \
    --rest-eta-neg 0.10 \
    --rest-horizon 2 \
    --rest-confidence-bins 0.33,0.66
```

---

## Output Files (written to `--output-dir`, default `output/`)

| File                    | Contents                                         |
|-------------------------|--------------------------------------------------|
| `summary.json`          | policy, total_cost, hits, misses, hit_rate       |
| `metrics.json`          | prediction error η, per-class weighted surprises |
| `per_step_decisions.csv`| one row per request: t, page, hit, cost, evicted |
| `rest_v1_diagnostics.json` | ReST trust/abstain decision log and context diagnostics (when `rest_v1`) |

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
- Twitter cache-trace (Twemcache / Pelikan) via manifest ingestion
- MetaKV via manifest/oracle-style ingestion
- MetaCDN via manifest/oracle-style ingestion
- CloudPhysics block I/O traces via manifest ingestion

Prepare datasets via:

```bash
python scripts/datasets/prepare_all.py --dataset <brightkite|citibike|spec_cpu2006|wiki2018|twemcache|metakv|metacdn|cloudphysics|all>
```

For the new production traces (`twemcache`, `metakv`, `metacdn`, `cloudphysics`), place local raw files under
`data/raw/<dataset>/` and provide `manifest.json` listing local files to ingest.

Processed traces are written under `data/processed/<dataset>/`.
See `docs/datasets.md` for source links, legal caveats, mapping assumptions, and exact commands.
