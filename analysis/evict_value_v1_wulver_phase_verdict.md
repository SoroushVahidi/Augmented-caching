# evict_value_v1 — first multi-family Wulver phase (verdict)

## What was run (this workspace)

1. **Dataset:** `scripts/build_evict_value_dataset_wulver_v1.py` with `analysis/wulver_trace_manifest_full.csv`, **subset** `--max-traces 3` (brightkite, citibike, wiki2018), `--capacities 64`, `--horizons 4,8,16`, `--max-requests-per-trace 3000`, output `data/derived/evict_value_v1_wulver_multi/`.
2. **Summaries:** `scripts/summarize_wulver_evict_value_dataset.py` → `analysis/evict_value_v1_wulver_dataset_summary.md` and `dataset_summary_extended.json`.
3. **Training:** `scripts/train_evict_value_wulver_v1.py` (Ridge, RandomForest, HistGradientBoosting only) → `analysis/evict_value_wulver_v1_train_metrics.json`, `analysis/evict_value_wulver_v1_model_comparison.csv`, `models/evict_value_wulver_v1_*.pkl`, best checkpoint `models/evict_value_wulver_v1_best.pkl`.
4. **Policy comparison:** intended via `scripts/run_policy_comparison_wulver_v1.py`; **ml_gate_v1/ml_gate_v2 are skipped** if `models/ml_gate_v1_random_forest.pkl` / `models/ml_gate_v2_random_forest.pkl` are absent. Re-run after training those models if you need them in the table.

## Dataset row / decision summary (this run)

| Metric | Value |
|--------|------:|
| Total rows | 849024 |
| Unique decisions | 13266 |
| Capacities | 64 only |
| Horizons | 4, 8, 16 |

Rows by split: train 285312, val 563712, **test 0** (no chunk landed in test for this trace subset).

Rows by family: brightkite 155904, citibike 129408, wiki2018 563712.

**Split skew:** under `trace_chunk`, **all brightkite and citibike rows fell in train** and **all wiki2018 rows in val** (see `family_x_split_rows` in `dataset_summary_extended.json`). Validation metrics are therefore **not a balanced multi-family mix** for this small run.

## Model comparison (H ∈ {4,8,16})

Source: `analysis/evict_value_wulver_v1_model_comparison.csv`.

| horizon | model | val_mae | val_rmse | val_top1 | val_mean_regret |
|--------:|--------|--------:|---------:|---------:|----------------:|
| 4 | ridge | 2.505 | 2.505 | 0.469 | 0.0 |
| 4 | random_forest | 2.481 | 2.488 | 0.951 | 0.0 |
| 4 | hist_gb | 2.408 | 2.409 | 0.009 | 0.0 |
| 8 | ridge | 5.131 | 5.131 | 0.469 | 0.0 |
| 8 | random_forest | 4.945 | 4.953 | 0.951 | 0.0 |
| 8 | hist_gb | 4.896 | 4.899 | 0.013 | 0.0 |
| 16 | ridge | 10.611 | 10.611 | 0.469 | 0.0 |
| 16 | random_forest | 10.004 | 10.017 | 0.002 | 0.0 |
| 16 | hist_gb | 9.792 | 9.803 | 0.004 | 0.0 |

**Selection rule (implemented):** minimize **validation `mean_regret_vs_oracle`**; tie-break with **lower val MAE**, then **val RMSE**.

On this run, **all `val_mean_regret` values are 0.0**, so the rule reduces to MAE: **best checkpoint = ridge at horizon 4** (`analysis/evict_value_wulver_v1_best_config.json`).

**Per-family validation:** metrics in the JSON show **wiki2018 only** on val for this run (consistent with split skew). This is **not** sufficient to claim multi-family generalization.

## Policy comparison

If `analysis/evict_value_wulver_v1_policy_comparison.md` is present, use it. Otherwise run:

```bash
PYTHONPATH=src python scripts/run_policy_comparison_wulver_v1.py \
  --trace-manifest analysis/wulver_trace_manifest_full.csv \
  --max-traces 7 \
  --capacities 64 \
  --max-requests-per-trace 3000 \
  --evict-value-model models/evict_value_wulver_v1_best.pkl
```

## wiki2018 caveat

wiki2018 traces here are **pageview-derived** (see `docs/datasets_wulver_trace_acquisition.md`), not original CDN object logs.

## Final classification: **MIXED / UNCERTAIN**

**Why not “clearly promising”**

- Only **3/7** families in the dataset job; full manifest run is required.
- **Train/val family split skew** makes val metrics **single-family (wiki2018)** in practice.
- **Regret = 0** on val for all models is suspicious (ties or insufficient diversity to discriminate); do not over-interpret ranking quality.
- **No test split** in this run for chunk-based assignment.

**Why not “not worth refinement”**

- Pipeline (shards, manifest, training script, policy harness) is in place for a **full** multi-family rerun.
- RandomForest achieves much higher **top1** than ridge/hist_gb on this slice — worth revisiting under a **properly mixed** val set before abandoning the direction.

## Evidence weaker than ideal (checklist)

1. Subset traces and capped request length.
2. Chunk split produced **no test** bucket and **family×split imbalance**.
3. Val regret degeneracy (all zeros).
4. ML-gate policies may be absent from policy tables unless models exist.
5. Resource limits on shared login nodes; use the Slurm script for full-scale work.
