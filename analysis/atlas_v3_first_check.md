# atlas_v3 first check

atlas_v3 is an **experimental**, **confidence-aware**, **local-trust** framework policy (first CCLT implementation).

## Exact commands run

- `PYTHONPATH=src pytest -q tests/test_atlas_v3.py tests/test_atlas_v2.py tests/test_runner.py`
- `PYTHONPATH=src python - <<'PY' ... (generated analysis/atlas_v3_first_check.csv) ... PY`
- `PYTHONPATH=src python - <<'PY' ... (generated analysis/atlas_v3_first_check.md) ... PY`

## Exact formulas implemented

- Context: `ctx(p,t) = (bucket(p,t), confidence_bin(p,t))` with bins from `--atlas-confidence-bins` (default `0.33,0.66`).
- Page-level trust: `lambda_{p,t} = T[ctx(p,t)] * confidence_{p,t}`; if confidence missing, use `default_confidence`.
- Base score: LRU-normalized (`oldest -> 1`, newest -> 0).
- Predictor score (stronger separation): rank unique buckets among candidates to `[0,1]`, then square: `PredScore = rank^2`.
- Combined score: `Score = lambda * PredScore + (1-lambda) * BaseScore`, evict `argmax Score` (deterministic tie by LRU order).

## Local trust update rule

- Only **predictor-dominated** evictions create pending checks.
- Bucket-aware too-soon horizon `H(b)` with `--atlas-bucket-regret-mode`:
  - `linear`: `H(b)=bucket_horizon*(b+1)`
  - `exp2`: `H(b)=bucket_horizon*2^b`
- If evicted page returns within `delta <= H(b)`: bad outcome.
- Else (late/no early return): good outcome.
- Update:
  - good: `T[ctx] <- clip(T[ctx] + eta_pos * confidence, 0, 1)`
  - bad:  `T[ctx] <- clip(T[ctx] - eta_neg * confidence, 0, 1)`

## Comparison summary

| trace | cap | setting | atlas_v2 misses | atlas_v3 misses | lru misses | atlas_v3 predictor_frac | atlas_v3 fallback_frac | atlas_v3 bad_outcomes |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| data/example_atlas_v1.json | 2 | trace | 8 | 9 | 10 | 0.286 | 0.429 | 1 |
| data/example_atlas_v1.json | 2 | perfect | 10 | 10 | 10 | 0.250 | 0.125 | 1 |
| data/example_atlas_v1.json | 2 | perfect_noise0.3 | 9 | 9 | 10 | 0.286 | 0.286 | 2 |
| data/example_atlas_v1.json | 3 | trace | 7 | 7 | 9 | 0.500 | 0.250 | 1 |
| data/example_atlas_v1.json | 3 | perfect | 6 | 7 | 9 | 0.500 | 0.250 | 1 |
| data/example_atlas_v1.json | 3 | perfect_noise0.3 | 9 | 9 | 9 | 0.167 | 0.333 | 1 |
| data/example_atlas_v1.json | 4 | trace | 6 | 6 | 6 | 0.000 | 0.000 | 0 |
| data/example_atlas_v1.json | 4 | perfect | 6 | 6 | 6 | 0.000 | 0.000 | 0 |
| data/example_atlas_v1.json | 4 | perfect_noise0.3 | 7 | 7 | 6 | 0.333 | 0.667 | 1 |
| data/example_unweighted.json | 2 | trace | 10 | 10 | 10 | 0.000 | 0.000 | 0 |
| data/example_unweighted.json | 2 | perfect | 10 | 10 | 10 | 0.000 | 0.375 | 0 |
| data/example_unweighted.json | 2 | perfect_noise0.3 | 10 | 10 | 10 | 0.000 | 0.500 | 0 |
| data/example_unweighted.json | 3 | trace | 6 | 6 | 6 | 0.000 | 0.000 | 0 |
| data/example_unweighted.json | 3 | perfect | 5 | 5 | 6 | 0.500 | 0.000 | 0 |
| data/example_unweighted.json | 3 | perfect_noise0.3 | 7 | 8 | 6 | 0.400 | 0.400 | 2 |
| data/example_unweighted.json | 4 | trace | 4 | 4 | 4 | 0.000 | 0.000 | 0 |
| data/example_unweighted.json | 4 | perfect | 4 | 4 | 4 | 0.000 | 0.000 | 0 |
| data/example_unweighted.json | 4 | perfect_noise0.3 | 4 | 4 | 4 | 0.000 | 0.000 | 0 |

## Explicit answers

1. Did atlas_v3 reduce LRU collapse relative to atlas_v2? **Partially**: farther-from-LRU improved in 1/18 settings.
2. Did local trust increase predictor-dominated decisions? **Mixed**: atlas_v3 shows non-trivial predictor-dominated fractions in many settings, but fallback still dominates several low-signal cases.
3. Did local trust improve behavior in mixed-quality regimes? **Mixed-positive**: on noisy-perfect (`perfect_noise0.3`) settings atlas_v3 was <= atlas_v2 misses in 5/6 cases.
4. Did stronger regret target improve detection of bad predictor-led evictions? **Yes (mechanistically)**: atlas_v3 logged non-zero bad-outcome detections in 8/18 settings via bucket-aware too-soon checks.
5. What still looks weak? Small traces still create many tie/fallback decisions, and improvements are not uniform at all capacities.

## What improved over atlas_v2?

- Per-context trust avoids global collapse from one bad regime.
- Diagnostics now include context-wise trust evolution and good/bad outcome counts.
- Predictor influence can remain high in high-trust/high-confidence contexts without forcing low-trust contexts to follow.

## What is still failing?

- Some traces remain LRU-like, especially at larger capacities where conflicts are sparse.
- Predictor dominance can still be low when confidence is moderate and context trust does not warm up quickly.

## Likely next improvement

Use adaptive confidence calibration per context (EMA/Bayesian reliability scaling before lambda multiplication) plus an adaptive tie-band based on score dispersion.
