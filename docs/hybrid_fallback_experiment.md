# Hybrid regime-aware / confidence-aware fallback experiment

## Why this experiment

Recent medium-scale experiments showed supervision changes (offline-teacher labels and pairwise/ranking-aware targets) did not improve downstream miss-rate over the current pointwise baseline. This experiment tests a different hypothesis: keep the learned candidate scorer, but add a simple fallback when decisions are low-confidence.

## What this experiment compares

Script: `scripts/run_hybrid_fallback_experiment.py`

Held-out test comparison includes:

1. `pointwise`: learned candidate scorer path (evict_value_v1-compatible lightweight surrogate scoring).
2. `hybrid`: pointwise + confidence-aware fallback trigger.
3. `lru`: pure fallback baseline.

## Trigger and fallback design

At each eviction decision (cache full + miss):

- Score candidates with the learned scorer.
- Compute `margin = score(top2) - score(top1)` where lower score is preferred.
- If `margin < threshold`, treat decision as fragile and fallback to **LRU** victim.
- Else use pointwise victim.

Primary fallback is LRU because it is robust, deterministic, already central in this repository, and easy to audit.

## Threshold selection protocol (no test tuning)

For each seed:

1. Split traces by hash(trace, seed) into train / val / test buckets.
2. Sweep a fixed threshold grid on **validation workloads only**.
3. Select threshold minimizing validation total misses (tie-breakers: lower trigger frequency, then smaller threshold).
4. Evaluate selected threshold on **held-out test workloads**.

No threshold tuning is done on test.

## Reported artifacts

The script writes these text artifacts to `--output-dir` (default: `analysis/hybrid_fallback_experiment`):

- `results.csv`
- `downstream_results.csv`
- `threshold_selection.csv`
- `trigger_analysis.csv`
- `decision_logs.csv`
- `summary.json`
- `report.md`

## Trigger fragility diagnostics

This experiment emits per-decision diagnostics and a simple fragility proxy:

- `victim_reused_within_window` in `decision_logs.csv`.
- Triggered vs non-triggered victim-reuse rates in `trigger_analysis.csv`.

Interpretation: if triggered decisions show higher victim-reuse rates, low-margin decisions are genuinely fragile.

## Run command

```bash
python scripts/run_hybrid_fallback_experiment.py \
  --trace-glob data/example_unweighted.json,data/example_atlas_v1.json,data/example_general_caching.json,data/example.json \
  --capacities 2,3,4,5 \
  --seeds 0,1,2,3,4,5 \
  --margin-thresholds 0.00,0.01,0.03,0.05,0.08,0.12 \
  --max-requests-per-trace 0 \
  --fragility-window 16 \
  --output-dir analysis/hybrid_fallback_experiment
```

## Limitations and next steps

- Current implementation uses the artifact-free evict_value_v1-compatible surrogate scorer for dependency-light local runs.
- Fragility analysis uses victim-reuse proxy instead of full oracle regret decomposition.
- Next clean extension: add normalized-margin trigger variant and optional second fallback baseline (`blind_oracle_lru_combiner`) under the same val-selected protocol.
