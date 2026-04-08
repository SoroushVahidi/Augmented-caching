# First Comparison Report

## Exact commands run

- `PYTHONPATH=src python scripts/run_first_comparison.py`

## Traces / capacities / policies

- Unweighted traces: `data/example_unweighted.json`, `data/example_atlas_v1.json`.
- Weighted trace: `data/example.json`.
- Capacities swept: 2, 3, 4.
- Unweighted policies: atlas_v1, lru, marker, blind_oracle, predictive_marker, trust_and_doubt.
- Weighted policies: lru, weighted_lru, advice_trusting, la_det.

## Unweighted results

| trace | cap | policy | setting | cost | misses | hit_rate | evictions |
|---|---:|---|---|---:|---:|---:|---:|
| data/example_unweighted.json | 2 | lru | default | 10.00 | 10 | 0.00% | 8 |
| data/example_unweighted.json | 2 | marker | default | 10.00 | 10 | 0.00% | 8 |
| data/example_unweighted.json | 2 | blind_oracle | default | 8.00 | 8 | 20.00% | 6 |
| data/example_unweighted.json | 2 | predictive_marker | default | 8.00 | 8 | 20.00% | 6 |
| data/example_unweighted.json | 2 | trust_and_doubt | default | 8.00 | 8 | 20.00% | 6 |
| data/example_unweighted.json | 2 | atlas_v1 | trace_conf | 10.00 | 10 | 0.00% | 8 |
| data/example_unweighted.json | 2 | atlas_v1 | perfect_conf_1.0 | 8.00 | 8 | 20.00% | 6 |
| data/example_unweighted.json | 2 | atlas_v1 | perfect_conf_0.5 | 10.00 | 10 | 0.00% | 8 |
| data/example_unweighted.json | 2 | atlas_v1 | perfect_conf_0.0 | 10.00 | 10 | 0.00% | 8 |
| data/example_unweighted.json | 2 | atlas_v1 | perfect_noise_0.1 | 10.00 | 10 | 0.00% | 8 |
| data/example_unweighted.json | 2 | atlas_v1 | perfect_noise_0.3 | 10.00 | 10 | 0.00% | 8 |
| data/example_unweighted.json | 3 | lru | default | 6.00 | 6 | 40.00% | 3 |
| data/example_unweighted.json | 3 | marker | default | 7.00 | 7 | 30.00% | 4 |
| data/example_unweighted.json | 3 | blind_oracle | default | 5.00 | 5 | 50.00% | 2 |
| data/example_unweighted.json | 3 | predictive_marker | default | 5.00 | 5 | 50.00% | 2 |
| data/example_unweighted.json | 3 | trust_and_doubt | default | 5.00 | 5 | 50.00% | 2 |
| data/example_unweighted.json | 3 | atlas_v1 | trace_conf | 6.00 | 6 | 40.00% | 3 |
| data/example_unweighted.json | 3 | atlas_v1 | perfect_conf_1.0 | 5.00 | 5 | 50.00% | 2 |
| data/example_unweighted.json | 3 | atlas_v1 | perfect_conf_0.5 | 6.00 | 6 | 40.00% | 3 |
| data/example_unweighted.json | 3 | atlas_v1 | perfect_conf_0.0 | 6.00 | 6 | 40.00% | 3 |
| data/example_unweighted.json | 3 | atlas_v1 | perfect_noise_0.1 | 6.00 | 6 | 40.00% | 3 |
| data/example_unweighted.json | 3 | atlas_v1 | perfect_noise_0.3 | 7.00 | 7 | 30.00% | 4 |
| data/example_unweighted.json | 4 | lru | default | 4.00 | 4 | 60.00% | 0 |
| data/example_unweighted.json | 4 | marker | default | 4.00 | 4 | 60.00% | 0 |
| data/example_unweighted.json | 4 | blind_oracle | default | 4.00 | 4 | 60.00% | 0 |
| data/example_unweighted.json | 4 | predictive_marker | default | 4.00 | 4 | 60.00% | 0 |
| data/example_unweighted.json | 4 | trust_and_doubt | default | 4.00 | 4 | 60.00% | 0 |
| data/example_unweighted.json | 4 | atlas_v1 | trace_conf | 4.00 | 4 | 60.00% | 0 |
| data/example_unweighted.json | 4 | atlas_v1 | perfect_conf_1.0 | 4.00 | 4 | 60.00% | 0 |
| data/example_unweighted.json | 4 | atlas_v1 | perfect_conf_0.5 | 4.00 | 4 | 60.00% | 0 |
| data/example_unweighted.json | 4 | atlas_v1 | perfect_conf_0.0 | 4.00 | 4 | 60.00% | 0 |
| data/example_unweighted.json | 4 | atlas_v1 | perfect_noise_0.1 | 4.00 | 4 | 60.00% | 0 |
| data/example_unweighted.json | 4 | atlas_v1 | perfect_noise_0.3 | 4.00 | 4 | 60.00% | 0 |
| data/example_atlas_v1.json | 2 | lru | default | 10.00 | 10 | 0.00% | 8 |
| data/example_atlas_v1.json | 2 | marker | default | 10.00 | 10 | 0.00% | 8 |
| data/example_atlas_v1.json | 2 | blind_oracle | default | 10.00 | 10 | 0.00% | 8 |
| data/example_atlas_v1.json | 2 | predictive_marker | default | 9.00 | 9 | 10.00% | 7 |
| data/example_atlas_v1.json | 2 | trust_and_doubt | default | 9.00 | 9 | 10.00% | 7 |
| data/example_atlas_v1.json | 2 | atlas_v1 | trace_conf | 8.00 | 8 | 20.00% | 6 |
| data/example_atlas_v1.json | 2 | atlas_v1 | perfect_conf_1.0 | 10.00 | 10 | 0.00% | 8 |
| data/example_atlas_v1.json | 2 | atlas_v1 | perfect_conf_0.5 | 10.00 | 10 | 0.00% | 8 |
| data/example_atlas_v1.json | 2 | atlas_v1 | perfect_conf_0.0 | 10.00 | 10 | 0.00% | 8 |
| data/example_atlas_v1.json | 2 | atlas_v1 | perfect_noise_0.1 | 9.00 | 9 | 10.00% | 7 |
| data/example_atlas_v1.json | 2 | atlas_v1 | perfect_noise_0.3 | 10.00 | 10 | 0.00% | 8 |
| data/example_atlas_v1.json | 3 | lru | default | 9.00 | 9 | 10.00% | 6 |
| data/example_atlas_v1.json | 3 | marker | default | 8.00 | 8 | 20.00% | 5 |
| data/example_atlas_v1.json | 3 | blind_oracle | default | 8.00 | 8 | 20.00% | 5 |
| data/example_atlas_v1.json | 3 | predictive_marker | default | 7.00 | 7 | 30.00% | 4 |
| data/example_atlas_v1.json | 3 | trust_and_doubt | default | 6.00 | 6 | 40.00% | 3 |
| data/example_atlas_v1.json | 3 | atlas_v1 | trace_conf | 7.00 | 7 | 30.00% | 4 |
| data/example_atlas_v1.json | 3 | atlas_v1 | perfect_conf_1.0 | 7.00 | 7 | 30.00% | 4 |
| data/example_atlas_v1.json | 3 | atlas_v1 | perfect_conf_0.5 | 7.00 | 7 | 30.00% | 4 |
| data/example_atlas_v1.json | 3 | atlas_v1 | perfect_conf_0.0 | 7.00 | 7 | 30.00% | 4 |
| data/example_atlas_v1.json | 3 | atlas_v1 | perfect_noise_0.1 | 8.00 | 8 | 20.00% | 5 |
| data/example_atlas_v1.json | 3 | atlas_v1 | perfect_noise_0.3 | 8.00 | 8 | 20.00% | 5 |
| data/example_atlas_v1.json | 4 | lru | default | 6.00 | 6 | 40.00% | 2 |
| data/example_atlas_v1.json | 4 | marker | default | 7.00 | 7 | 30.00% | 3 |
| data/example_atlas_v1.json | 4 | blind_oracle | default | 7.00 | 7 | 30.00% | 3 |
| data/example_atlas_v1.json | 4 | predictive_marker | default | 5.00 | 5 | 50.00% | 1 |
| data/example_atlas_v1.json | 4 | trust_and_doubt | default | 5.00 | 5 | 50.00% | 1 |
| data/example_atlas_v1.json | 4 | atlas_v1 | trace_conf | 6.00 | 6 | 40.00% | 2 |
| data/example_atlas_v1.json | 4 | atlas_v1 | perfect_conf_1.0 | 6.00 | 6 | 40.00% | 2 |
| data/example_atlas_v1.json | 4 | atlas_v1 | perfect_conf_0.5 | 6.00 | 6 | 40.00% | 2 |
| data/example_atlas_v1.json | 4 | atlas_v1 | perfect_conf_0.0 | 6.00 | 6 | 40.00% | 2 |
| data/example_atlas_v1.json | 4 | atlas_v1 | perfect_noise_0.1 | 7.00 | 7 | 30.00% | 3 |
| data/example_atlas_v1.json | 4 | atlas_v1 | perfect_noise_0.3 | 7.00 | 7 | 30.00% | 3 |

## Weighted results

| trace | cap | policy | setting | cost | misses | hit_rate | evictions |
|---|---:|---|---|---:|---:|---:|---:|
| data/example.json | 2 | lru | default | 19.00 | 10 | 0.00% | 8 |
| data/example.json | 2 | weighted_lru | default | 15.00 | 9 | 10.00% | 7 |
| data/example.json | 2 | advice_trusting | default | 17.00 | 8 | 20.00% | 6 |
| data/example.json | 2 | la_det | default | 14.00 | 8 | 20.00% | 6 |
| data/example.json | 3 | lru | default | 14.00 | 6 | 40.00% | 3 |
| data/example.json | 3 | weighted_lru | default | 9.00 | 5 | 50.00% | 2 |
| data/example.json | 3 | advice_trusting | default | 10.00 | 5 | 50.00% | 2 |
| data/example.json | 3 | la_det | default | 9.00 | 5 | 50.00% | 2 |
| data/example.json | 4 | lru | default | 8.00 | 4 | 60.00% | 0 |
| data/example.json | 4 | weighted_lru | default | 8.00 | 4 | 60.00% | 0 |
| data/example.json | 4 | advice_trusting | default | 8.00 | 4 | 60.00% | 0 |
| data/example.json | 4 | la_det | default | 8.00 | 4 | 60.00% | 0 |

## atlas_v1 diagnostic summary

| trace | cap | setting | avg_lambda | low_conf_frac | fallback_frac | mismatch_proxy | match_lru | match_blind_oracle |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| data/example_unweighted.json | 2 | trace_conf | 0.500 | 0.000 | 1.000 | 0.000 | 1.000 | 0.125 |
| data/example_unweighted.json | 2 | perfect_conf_1.0 | 1.000 | 0.000 | 0.000 | 0.000 | 0.333 | 0.167 |
| data/example_unweighted.json | 2 | perfect_conf_0.5 | 0.500 | 0.000 | 1.000 | 0.000 | 1.000 | 0.125 |
| data/example_unweighted.json | 2 | perfect_conf_0.0 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.125 |
| data/example_unweighted.json | 2 | perfect_noise_0.1 | 0.500 | 0.000 | 1.000 | 0.000 | 1.000 | 0.125 |
| data/example_unweighted.json | 2 | perfect_noise_0.3 | 0.500 | 0.000 | 1.000 | 0.000 | 1.000 | 0.125 |
| data/example_unweighted.json | 3 | trace_conf | 0.500 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| data/example_unweighted.json | 3 | perfect_conf_1.0 | 1.000 | 0.000 | 0.000 | 0.000 | 0.500 | 0.000 |
| data/example_unweighted.json | 3 | perfect_conf_0.5 | 0.500 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| data/example_unweighted.json | 3 | perfect_conf_0.0 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| data/example_unweighted.json | 3 | perfect_noise_0.1 | 0.500 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| data/example_unweighted.json | 3 | perfect_noise_0.3 | 0.500 | 0.000 | 0.750 | 0.000 | 0.500 | 0.000 |
| data/example_unweighted.json | 4 | trace_conf | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| data/example_unweighted.json | 4 | perfect_conf_1.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| data/example_unweighted.json | 4 | perfect_conf_0.5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| data/example_unweighted.json | 4 | perfect_conf_0.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| data/example_unweighted.json | 4 | perfect_noise_0.1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| data/example_unweighted.json | 4 | perfect_noise_0.3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| data/example_atlas_v1.json | 2 | trace_conf | 0.733 | 0.000 | 0.167 | 0.600 | 0.500 | 0.167 |
| data/example_atlas_v1.json | 2 | perfect_conf_1.0 | 0.675 | 0.000 | 0.250 | 0.000 | 0.750 | 0.125 |
| data/example_atlas_v1.json | 2 | perfect_conf_0.5 | 0.675 | 0.000 | 0.250 | 0.000 | 0.750 | 0.125 |
| data/example_atlas_v1.json | 2 | perfect_conf_0.0 | 0.675 | 0.000 | 0.250 | 0.000 | 0.750 | 0.125 |
| data/example_atlas_v1.json | 2 | perfect_noise_0.1 | 0.664 | 0.000 | 0.429 | 0.000 | 0.714 | 0.286 |
| data/example_atlas_v1.json | 2 | perfect_noise_0.3 | 0.675 | 0.000 | 0.250 | 0.200 | 0.625 | 0.625 |
| data/example_atlas_v1.json | 3 | trace_conf | 0.658 | 0.000 | 0.500 | 0.600 | 0.500 | 0.000 |
| data/example_atlas_v1.json | 3 | perfect_conf_1.0 | 0.658 | 0.000 | 0.500 | 0.000 | 0.500 | 0.000 |
| data/example_atlas_v1.json | 3 | perfect_conf_0.5 | 0.658 | 0.000 | 0.500 | 0.000 | 0.500 | 0.000 |
| data/example_atlas_v1.json | 3 | perfect_conf_0.0 | 0.658 | 0.000 | 0.500 | 0.000 | 0.500 | 0.000 |
| data/example_atlas_v1.json | 3 | perfect_noise_0.1 | 0.627 | 0.000 | 0.600 | 0.000 | 0.800 | 0.000 |
| data/example_atlas_v1.json | 3 | perfect_noise_0.3 | 0.653 | 0.000 | 0.200 | 0.200 | 0.400 | 0.000 |
| data/example_atlas_v1.json | 4 | trace_conf | 0.637 | 0.000 | 0.000 | 0.600 | 1.000 | 0.000 |
| data/example_atlas_v1.json | 4 | perfect_conf_1.0 | 0.637 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| data/example_atlas_v1.json | 4 | perfect_conf_0.5 | 0.637 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| data/example_atlas_v1.json | 4 | perfect_conf_0.0 | 0.637 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| data/example_atlas_v1.json | 4 | perfect_noise_0.1 | 0.625 | 0.000 | 0.333 | 0.000 | 0.333 | 0.000 |
| data/example_atlas_v1.json | 4 | perfect_noise_0.3 | 0.617 | 0.000 | 0.333 | 0.200 | 0.333 | 0.333 |

## What seems to be going on?

- `data/example_atlas_v1.json` cap=2: best misses=8 by atlas_v1:trace_conf; atlas(trace_conf) misses=8.
- `data/example_atlas_v1.json` cap=3: best misses=6 by trust_and_doubt:default; atlas(trace_conf) misses=7.
- `data/example_atlas_v1.json` cap=4: best misses=5 by predictive_marker:default, trust_and_doubt:default; atlas(trace_conf) misses=6.
- `data/example_unweighted.json` cap=2: best misses=8 by blind_oracle:default, predictive_marker:default, trust_and_doubt:default, atlas_v1:perfect_conf_1.0; atlas(trace_conf) misses=10.
- `data/example_unweighted.json` cap=3: best misses=5 by blind_oracle:default, predictive_marker:default, trust_and_doubt:default, atlas_v1:perfect_conf_1.0; atlas(trace_conf) misses=6.
- `data/example_unweighted.json` cap=4: best misses=4 by lru:default, marker:default, blind_oracle:default, predictive_marker:default, trust_and_doubt:default, atlas_v1:trace_conf, atlas_v1:perfect_conf_1.0, atlas_v1:perfect_conf_0.5, atlas_v1:perfect_conf_0.0, atlas_v1:perfect_noise_0.1, atlas_v1:perfect_noise_0.3; atlas(trace_conf) misses=4.
- Perfect buckets only help atlas_v1 consistently when confidence is high (`perfect_conf_1.0`).
- Higher bucket noise (0.3) generally increases atlas_v1 misses and fallback-dominated decisions.

## Where atlas_v1 looks weak

- High match-to-LRU fractions in several settings suggest trust blending often defaults to recency, not prediction-led choices.
- In low-confidence settings, atlas_v1 provides little separation from LRU and rarely tracks blind_oracle choices.
- BlindOracle/PredictiveMarker or TRUST&DOUBT can dominate atlas_v1 on the current toy traces.
- On traces without strong per-page confidence variation, lambda has limited leverage over outcomes.

## Most likely next improvement

1. Calibrate lambda using online reliability (e.g., shrink confidence when mismatch proxy rises) rather than static per-page values only.
2. Improve PredScore normalization to preserve bucket separation even when candidate bucket range is narrow.
3. Add explicit tie-breaking toward predictor when confidence is high to avoid accidental LRU collapse on score ties.
4. Add more diverse tiny traces where predictor and recency conflict frequently to stress trust logic.
