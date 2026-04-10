# Learning-Augmented Caching (`lafc`)

A research codebase for learning-augmented caching algorithms, robust baselines, and manuscript-support experiments.

This repository includes:
- faithful and robust policy baselines from the LA-caching literature,
- experimental learned eviction policies and guard wrappers,
- dataset-preparation pipelines for public and manifest-based traces,
- reproducible analysis scripts that emit text artifacts (`.csv`, `.json`, `.md`).

## Project status (conservative snapshot)

- **Most stable, manuscript-safe references:** classical baselines (`lru`, `marker`, `predictive_marker`), robust combiners (`robust_ftp_d_marker`, `blind_oracle_lru_combiner`), offline references (`offline_belady`, `offline_general_caching_approx`).
- **Main learned line in this repo:** `evict_value_v1` (candidate-level scoring) plus guarded/fallback variants.
- **Research-active exploratory lines:** pairwise/ranking supervision, sentinel mechanisms, and theorem-development notes under `docs/pairwise_*`.

For evidence-strength caveats and what is still open, see `docs/manuscript_evidence_map.md` and `docs/manuscript_open_questions.md`.

---

## Installation

```bash
pip install -e ".[dev]"
```

---

## Quick start

### Run a baseline policy

```bash
python -m lafc.runner.run_policy \
  --policy predictive_marker \
  --trace data/example_unweighted.json \
  --capacity 3
```

### Run a robust combiner baseline

```bash
python -m lafc.runner.run_policy \
  --policy robust_ftp_d_marker \
  --trace data/example_unweighted.json \
  --capacity 3 \
  --derive-predicted-caches
```

### Run the current main learned policy (`evict_value_v1`)

```bash
python scripts/build_evict_value_dataset_v1.py --max-rows 200000
python scripts/train_evict_value_v1.py --horizon 8
python scripts/run_evict_value_v1_first_check.py
```

---

## Policy families

### Literature baselines and robust references

- `lru`, `weighted_lru`, `advice_trusting`, `la_det`
- `marker`, `blind_oracle`, `predictive_marker`
- `adaptive_query` (`parsimonious_caching` alias)
- `trust_and_doubt`
- `robust_ftp_d_marker` (`robust_ftp` alias)
- `blind_oracle_lru_combiner`
- `offline_belady`, `offline_general_caching_approx`

### Experimental policies

- `atlas_v1`, `atlas_v2`, `atlas_v3`
- `atlas_cga_v1`, `atlas_cga_v2`
- `rest_v1`
- `ml_gate_v1`, `ml_gate_v2`
- `evict_value_v1`, `evict_value_v1_guarded`, `evict_value_pairwise_v1`
- `sentinel_robust_tripwire_v1`, `sentinel_budgeted_guard_v2`

See `docs/baselines.md` and `docs/framework.md` for algorithm notes and positioning.

---

## Datasets

Supported dataset families include BrightKite, CitiBike, SPEC CPU2006, wiki2018, and manifest-based production-style traces (`twemcache`, `metakv`, `metacdn`, `cloudphysics`).

Prepare all or one dataset:

```bash
python scripts/datasets/prepare_all.py --dataset <brightkite|citibike|spec_cpu2006|wiki2018|twemcache|metakv|metacdn|cloudphysics|all>
```

- Raw inputs: `data/raw/<dataset>/`
- Processed outputs: `data/processed/<dataset>/`
- Format and legal/source notes: `docs/datasets.md`

---

## Reproducing main experiment families

### A) `evict_value_v1` first check

```bash
python scripts/build_evict_value_dataset_v1.py --max-rows 200000
python scripts/train_evict_value_v1.py --horizon 8
python scripts/run_evict_value_v1_first_check.py
```

Outputs (default):
- `analysis/evict_value_v1_first_check.csv`
- `analysis/evict_value_v1_first_check.md`
- `analysis/evict_value_v1_metrics.json`
- `analysis/evict_value_v1_model_comparison.csv`

### B) Offline-teacher vs heuristic supervision

```bash
python scripts/run_offline_teacher_vs_heuristic_experiment.py \
  --trace-glob "data/example_*.json,data/example_general_caching.json" \
  --capacities 2,3 \
  --horizon 12 \
  --output-dir analysis/offline_teacher_vs_heuristic
```

For the stronger controlled run, see `docs/offline_teacher_vs_heuristic_mediumscale.md`.

### C) Pairwise vs pointwise supervision

```bash
python scripts/run_pairwise_vs_pointwise_experiment.py \
  --output-dir analysis/pairwise_vs_pointwise
```

Interpret conservatively; see `docs/pairwise_vs_pointwise_experiment.md` and `docs/manuscript_evidence_map.md`.

### D) Sentinel/guard refinement

```bash
python scripts/run_sentinel_budgeted_guard_v2_eval.py
python scripts/run_sentinel_budgeted_guard_v2_ablation.py
```

---

## Output conventions

Default output root is `analysis/` for manuscript-support artifacts and `output/` for ad hoc runs.

Typical files:
- `summary.json` (aggregate metrics)
- `report.md` (human-readable interpretation)
- `*.csv` (slice-level tables)
- `*_summary.json` (compact machine-readable summary)

See `analysis/README.md` for canonical vs legacy/exploratory organization.

---

## Repository guide

- High-level map: `docs/repo_map.md`
- Analysis artifact organization: `analysis/README.md`
- Script organization and naming conventions: `scripts/README.md`

---

## Testing

```bash
pytest tests/ -v
```

