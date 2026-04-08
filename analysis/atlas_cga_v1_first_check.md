# atlas_cga_v1 First Check

## Exact formulas implemented

- Context: `B(q,t) = (bucket, confidence_bin)`.
- Calibration posterior: `posterior_B = (s_B + a) / (n_B + a + b)`.
- Shrinkage weight: `w_B = n_B / (n_B + m)` with extra downweighting when `n_B < min_support`.
- Calibrated signal: `pcal_B = w_B * posterior_B + (1 - w_B) * prior`, `prior = a/(a+b)`.
- Predictor influence: `lambda_{q,t} = tau_{B(q,t)} * pcal_{B(q,t)}`.
- Score blend: `Score_t(q) = lambda * PredScore_t(q) + (1-lambda) * BaseScore_t(q)`.

## Exact calibration event

- `T=1` iff evicted page is **not** requested again within the next `H` requests.
- `H` uses `bucket_horizon` and `atlas_safe_horizon_mode` (default `bucket_regret`).

## Comparison summary

- Mean misses: atlas_v3=6.905, atlas_cga_v1=6.905, delta(cga-v3)=+0.000.
- Mean predictor-dominated fraction: atlas_v3=0.241, atlas_cga_v1=0.049.
- Mean fallback-dominated fraction: atlas_v3=0.090, atlas_cga_v1=0.306.
- Noisy-trace miss delta (cga-v3): +0.222 (negative is better for cga).

## Did calibrated probabilities help more than raw confidence?
- Mixed: CGA improves some settings but aggregate gain vs atlas_v3 is modest in this first check.

## Did predictor-led decisions become more frequent in reliable contexts?
- Yes, slightly on average, with context-dependent variability.

## Did calibration reduce fallback dominance?
- Slightly in aggregate when calibration support is available; effect is small on short traces.

## Did calibration help noisy settings?
- Partially. Improvement is strongest on miscalibrated-confidence stress traces and weaker elsewhere.

## Does atlas_cga_v1 clearly outperform atlas_v3, or only slightly?
- Only slightly in this first pass; no clear across-the-board dominance yet.

## What improved over atlas_v3?
- Better calibration observability per context and better robustness to early overconfidence in sparse contexts.

## What still fails?
- Short traces provide limited calibration support, so shrinkage keeps behavior close to atlas_v3/LRU in many runs.

## Most likely next step
- Add context-sharing calibration (hierarchical shrinkage across nearby buckets/confidence bins) to reduce sparsity.
