# ReST v1 First Check

## Exact formulas / update rules

- Context: `ctx_t = (bucket(request_t), confidence_bin(request_t))`.
- Gate state: per-context trust `G[ctx] in [0,1]`.
- Deterministic decision rule: TRUST iff `G[ctx_t] >= theta`, else ABSTAIN to LRU (`theta=0.5`).
- TRUST eviction expert: atlas_v3-style predictor score: bucket rank normalized to [0,1], squared.
- ABSTAIN expert: pure LRU eviction.
- Delayed feedback horizon `H=2` requests after trusted eviction.
- If evicted page returns within `H`: bad trust outcome, `G[ctx] <- clip01(G[ctx] - eta_neg)`.
- Else (no return within `H`): good trust outcome, `G[ctx] <- clip01(G[ctx] + eta_pos)`.
- Used parameters: `G0=0.5`, `eta_pos=0.05`, `eta_neg=0.10`, bins=`0.33,0.66`.

## Comparison summary

- Mean misses (all traces/capacities): rest_v1=6.476, atlas_v3=6.905, atlas_cga_v1=6.905, atlas_cga_v2=6.952, predictive_marker=6.333.
- Decisive predictor-use proxy: rest_v1 trust_coverage=0.620; atlas_v3 predictor_dominated=0.241; cga_v1=0.049; cga_v2=0.052.
- Stress-only mean misses: rest_v1=6.400, atlas_v3=6.867, cga_v1=6.733, cga_v2=6.733.

## Q1. Does selective trust reduce fallback dominance more than atlas/cga?
- In this first run, selective trust changes fallback behavior via hard gating (explicit TRUST/ABSTAIN), but aggregate dominance reduction is mixed and trace-dependent.
## Q2. Does it create more decisive predictor use in good contexts?
- Yes in mechanism (binary gate by context), with trust concentrated in some contexts; aggregate gains are modest.
## Q3. Is it more robust than naive predictor-following?
- Generally yes: abstention-to-LRU limits damage relative to always predictor-following baselines on adverse regimes.
## Q4. Does it beat atlas_v3 / atlas_cga on stress traces?
- Mixed in this first check; no universal dominance across all stress traces/capacities.
## Q5. Main remaining bottleneck after this pivot?
- Sparse feedback per context (few trusted updates) keeps trust adaptation slow and conservative.

## What improved over atlas/cga?
- Architectural clarity: explicit abstention gate and direct regret-like trust updates, decoupled from calibration weighting.
## What still fails?
- Short traces and frequent regime shifts still under-inform context trust quickly enough.
## Is this pivot more promising than continued calibration refinement?
- Tentatively yes as a direction: it explores a genuinely different decision architecture, though current gains are still modest.
