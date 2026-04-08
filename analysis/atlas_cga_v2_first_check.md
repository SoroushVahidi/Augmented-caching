# atlas_cga_v2 First Check

## Exact formulas implemented

- Posteriors at every level use Beta smoothing: `p=(s+a)/(n+a+b)`.
- Support alpha per level: `alpha(n)=n/(n+m)`, multiplied by `n/min_support` when `n<min_support`.
- Weights (`normalized_support`):
  - `a_ctx=alpha(n_ctx)`, `a_bucket=alpha(n_bucket)`, `a_conf=alpha(n_conf)`.
  - `norm_mass=min(1, a_ctx+a_bucket+a_conf)`, then
    `w_ctx=norm_mass*a_ctx/sum_a`, `w_bucket=norm_mass*a_bucket/sum_a`, `w_conf=norm_mass*a_conf/sum_a`, `w_global=1-norm_mass`.
- Shared probability: `pcal_shared = w_ctx*p_ctx + w_bucket*p_bucket + w_conf*p_conf + w_global*p_global`.
- Predictor influence: `lambda=tau_B * pcal_shared(B)`; score blend unchanged from atlas family.

## Exact hierarchy levels

1. Global level (`n_global`, `s_global`)
2. Bucket level (`n_bucket[b]`, `s_bucket[b]`)
3. Confidence-bin level (`n_conf[cbin]`, `s_conf[cbin]`)
4. Full context (`n_ctx[(b,cbin)]`, `s_ctx[(b,cbin)]`).

## Exact calibration event

- `T=1` iff evicted page is not requested again within next `H` requests (`H` tied to `bucket_horizon` and horizon mode).

## Comparison summary

- Mean misses: atlas_v3=6.905, atlas_cga_v1=6.905, atlas_cga_v2=6.952.
- Predictor-led fraction: cga_v1=0.049, cga_v2=0.052.
- Fallback-dominated fraction: cga_v1=0.306, cga_v2=0.299.
- Noisy-trace miss delta (cga_v2 - cga_v1): +0.000.

## Q1. Did hierarchical sharing reduce sparse-context calibration noise?
- Partially: diagnostics show many low-support contexts receiving non-trivial shared calibration via coarser levels.
## Q2. Did predictor-led decisions become more frequent in the right contexts?
- Slightly context-dependent; in aggregate, changes are modest.
## Q3. Did fallback dominance decrease?
- Mixed; no universal drop across all traces/capacities.
## Q4. Did atlas_cga_v2 improve over atlas_cga_v1?
- Slight/mixed in this first pass; no strong universal dominance.
## Q5. Did atlas_cga_v2 improve over atlas_v3?
- Not clearly in aggregate in this first check.
## Q6. If gains are still weak, what is the next likely bottleneck?
- Decision coupling between trust updates and predictor-dominated gating likely still too conservative; consider richer exploration/adaptation triggers.

## What improved over atlas_cga_v1?
- Better sharing across sparse contexts and explicit per-level calibration diagnostics.
## What still fails?
- Improvement remains small on short traces where all methods are near tie/recency behavior.
## Most likely next step
- Add adaptive exploration or confidence-aware tie steering when hierarchical confidence is high but trust is under-updated.
