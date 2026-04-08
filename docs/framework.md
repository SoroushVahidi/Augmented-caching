# Experimental Framework: Learning-Augmented Caching (Calibration + Selective Trust)

## Status

This framework is **experimental** and currently implements seven framework policies:
- `atlas_v1` (first version),
- `atlas_v2` (confidence-aware, dynamically trust-adaptive iteration),
- `atlas_v3` (confidence-aware local-trust iteration; first CCLT version),
- `atlas_cga_v1` (confidence-aware local-trust + online calibration-guided scaling),
- `atlas_cga_v2` (hierarchical/context-sharing calibration refinement of CGA),
- `rest_v1` (ReST selective-trust/abstention pivot with regret-like context updates).
- `evict_value_v1` (direct candidate-centric eviction-value predictor).
It is intended for empirical comparisons against existing baselines in this repository.

**Important:** this implementation does **not** claim a proved theorem or competitive guarantee.

## Goal

At each eviction, combine:
1. a prediction-derived score,
2. a confidence-aware trust weight,
3. a robust fallback score from LRU.

## ATLAS v1 (implemented baseline framework variant)

Scope in v1:
- unweighted paging only,
- cache capacity `k` pages,
- unit page size,
- unit miss cost,
- bucketed "needed soon" predictions,
- optional confidence in `[0,1]`,
- deterministic decisions,
- fallback toward LRU via confidence-weighted blending.

### Score used by `atlas_v1`

For each cached page `i` at time `t`:

- `PredScore_t(i)`: normalized from bucket (`larger bucket => more evictable`),
- `BaseScore_t(i)`: normalized LRU eviction score (`older => more evictable`),
- `lambda_{i,t}`: trust weight.

ATLAS v1 computes:

```text
Score_t(i) = lambda_{i,t} * PredScore_t(i)
           + (1 - lambda_{i,t}) * BaseScore_t(i)
```

Then evicts `argmax_i Score_t(i)`.

### Trust rule in v1

```text
lambda_{i,t} = confidence_{i,t} if confidence is present
               default_confidence otherwise
```

Default CLI value: `--default-confidence 0.5`.

## Input format for bucket/confidence hints

Preferred JSON extension:

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

`bucket` uses integer semantics where larger means less urgent / more evictable.
`confidence` is optional and clamped to `[0,1]`.

## Extra experimental options

- `--bucket-source trace|perfect`: use trace buckets or generate perfect buckets from actual next arrivals.
- `--bucket-horizon`: controls perfect-bucket discretization.
- `--bucket-noise-prob` and `--bucket-noise-seed`: optional synthetic corruption for bucketed hints.

## What is not yet implemented

- weighted paging integration,
- file caching / non-unit sizes,
- theoretical bounds for this framework variant,
- any claim of production readiness.

---

## ATLAS v2 (experimental confidence-aware + dynamic trust adaptation)

`atlas_v2` keeps the same blended score structure but introduces a global online trust
multiplier `gamma_t` and improved predictor-score normalization.

### Score used by `atlas_v2`

For cached page `i` at time `t`:

```text
Score_t(i) = lambda_{i,t} * PredScore_t(i)
           + (1 - lambda_{i,t}) * BaseScore_t(i)
```

Where:

- `BaseScore_t(i)` is the same normalized LRU score as v1 (`oldest -> 1`, newest -> `0`).
- `lambda_{i,t}` uses confidence and global trust:

```text
lambda_{i,t} = gamma_t * confidence_{i,t}           (if confidence is present)
lambda_{i,t} = gamma_t * default_confidence         (otherwise)
```

### `PredScore_t(i)` normalization in v2

Given candidate bucket `b_i`, candidate min/max `b_min, b_max`, and range `R = b_max - b_min`:

```text
raw_i    = (b_i - b_min) / max(1, R)
rank_i   = normalized rank of b_i among unique candidate bucket values in [0,1]
spread   = R / (R + 1)
PredScore_t(i) = spread * raw_i + (1 - spread) * rank_i
```

This is monotone in bucket value (larger bucket => more evictable) and keeps predictor signal
active even when the bucket spread is narrow.

### Dynamic trust (`gamma_t`) update

`atlas_v2` tracks a rolling mismatch proxy over the latest `W` resolved
predictor-dominated events:

```text
E_t = rolling mismatch rate in [0,1]
gamma_{t+1} = clip((1-rho) * gamma_t + rho * (1 - E_t), 0, 1)
```

### Mismatch proxy used (online-safe delayed bookkeeping)

For a predictor-dominated eviction of page `p` with bucket hint `b_p`:

- keep the event pending for up to `mismatch_threshold` future requests,
- if `p` is requested again within that window **and** `b_p >= soon_bucket_cutoff`
  (predicted “not soon”), record mismatch `1`,
- otherwise record mismatch `0`.

This uses only information available online when future requests actually arrive.

### Tie-breaking

When combined scores are tied (or nearly tied), tie-break is deterministic:
- if mean tie-set `lambda` is high, prefer predictor-favored candidate,
- otherwise prefer fallback (LRU)-favored candidate.

---

## Experimental honesty notes

- `atlas_v1`, `atlas_v2`, `atlas_v3`, `atlas_cga_v1`, `atlas_cga_v2`, and `rest_v1` are all **experimental**.
- `atlas_v2` is **confidence-aware** and **dynamically trust-adaptive**.
- `atlas_v3` is **confidence-aware**, **local-trust**, and context-adaptive.
- `atlas_cga_v1` is **experimental**, **calibration-guided**, **confidence-aware**, and
  built directly on the local-trust framework from `atlas_v3`.
- `atlas_cga_v2` is **experimental**, **hierarchical calibration/context-sharing**, and
  still built on the same local-trust score family.
- `rest_v1` is **experimental**, **selective-trust/abstention-style**, and a
  structural pivot away from calibration-heavy atlas refinement.
- Neither variant is presented as theoretically proven in this repository.

---

## ATLAS v3 (experimental CCLT v1: confidence-calibrated local trust)

`atlas_v3` replaces a single global trust scalar with local trust by context:

```text
ctx(i,t) = (bucket_i, confidence_bin_i)
T[ctx] in [0,1]
lambda_{i,t} = T[ctx(i,t)] * confidence_{i,t}
```

and keeps the blended eviction score:

```text
Score_t(i) = lambda_{i,t} * PredScore_t(i)
           + (1 - lambda_{i,t}) * BaseScore_t(i)
```

### Predictor score in v3 (stronger separation)

`atlas_v3` uses rank-based bucket scoring and squares the normalized rank:

```text
rank_i = normalized rank of bucket_i among unique candidate buckets in [0,1]
PredScore_t(i) = rank_i^2
```

This keeps predictor ordering active even when bucket values are numerically close.

### Bucket-aware rapid-regret target in v3

Only predictor-dominated evictions create pending checks.
For evicted page with bucket `b`, define tolerated return horizon `H(b)`:

- `linear` mode: `H(b) = bucket_horizon * (b + 1)`
- `exp2` mode: `H(b) = bucket_horizon * 2^b`

Outcome:
- bad if evicted page returns in `delta <= H(b)`,
- good otherwise (including no early return until expiration).

Local trust update:

```text
T[ctx] <- clip(T[ctx] + eta_pos * confidence, 0, 1)   (good)
T[ctx] <- clip(T[ctx] - eta_neg * confidence, 0, 1)   (bad)
```

with `eta_neg >= eta_pos`.

---

## ATLAS CGA v1 (experimental calibration-guided local trust)

`atlas_cga_v1` keeps `atlas_v3`'s context-local trust table and score family, and adds
an online calibration layer for safe-to-evict probabilities.

Context:

```text
B(i,t) = (bucket_i, confidence_bin_i)   (same context family as atlas_v3)
```

Calibration event:

```text
T = 1  iff evicted page does NOT return within next H requests
T = 0  otherwise
```

`H` is controlled by `bucket_horizon` and `--atlas-safe-horizon-mode`.

Per-context online statistics:

```text
n_B = number of resolved calibration outcomes
s_B = number of safe outcomes (T=1)
```

Smoothed calibration estimate and shrinkage:

```text
posterior_B = (s_B + a) / (n_B + a + b)
w_B         = n_B / (n_B + m), with extra downweighting below min-support
pcal_B      = w_B * posterior_B + (1 - w_B) * prior
prior       = a / (a + b)
```

Predictor influence and blended score:

```text
lambda_{i,t} = tau_{B(i,t)} * pcal_{B(i,t)}
Score_t(i)   = lambda_{i,t} * PredScore_t(i)
             + (1 - lambda_{i,t}) * BaseScore_t(i)
```

Local trust updates keep atlas_v3's good/bad logic, but update magnitude uses
the calibrated signal (not raw confidence).

---

## ATLAS CGA v2 (experimental hierarchical context-sharing calibration)

`atlas_cga_v2` keeps all CGA-v1 mechanics (safe-event definition, local trust, score blend)
but replaces context-isolated calibration with hierarchical sharing.

Hierarchy levels:

```text
L0: global
L1: bucket-only and confidence-bin-only
L2: full context (bucket, confidence_bin)
```

Each level uses Beta-smoothed posterior:

```text
p = (s + a) / (n + a + b)
```

Support-aware weight rule (`normalized_support`):

```text
alpha(n) = n / (n + m), additionally scaled by n/min_support when n < min_support

a_ctx    = alpha(n_ctx)
a_bucket = alpha(n_bucket)
a_conf   = alpha(n_conf)
norm_mass = min(1, a_ctx + a_bucket + a_conf)

w_ctx    = norm_mass * a_ctx    / (a_ctx + a_bucket + a_conf)
w_bucket = norm_mass * a_bucket / (a_ctx + a_bucket + a_conf)
w_conf   = norm_mass * a_conf   / (a_ctx + a_bucket + a_conf)
w_global = 1 - norm_mass
```

Shared calibration probability:

```text
pcal_shared(B) = w_ctx * p_ctx(B)
               + w_bucket * p_bucket(b)
               + w_conf * p_conf(cbin)
               + w_global * p_global
```

Predictor influence remains:

```text
lambda_{i,t} = tau_{B(i,t)} * pcal_shared(B(i,t))
Score_t(i)   = lambda_{i,t} * PredScore_t(i)
             + (1 - lambda_{i,t}) * BaseScore_t(i)
```

---

## ReST v1 (experimental selective trust / abstention pivot)

`rest_v1` changes the decision architecture from continuous blending to explicit gating:

- **TRUST mode**: use a predictor-driven eviction expert (atlas_v3-style bucket-rank squared score).
- **ABSTAIN mode**: use robust LRU eviction.

Default context:

```text
ctx_t = (bucket(request_t), confidence_bin(request_t))
G[ctx] in [0,1]
```

Deterministic gate:

```text
if G[ctx_t] >= theta: TRUST
else:                 ABSTAIN (LRU)
```

with default `theta = 0.5`.

Delayed online-safe trust feedback with horizon `H`:

- if a TRUST-evicted page returns within `H` requests: bad outcome,
- otherwise: good outcome.

Update:

```text
G[ctx] <- clip(G[ctx] + eta_pos, 0, 1)   (good)
G[ctx] <- clip(G[ctx] - eta_neg, 0, 1)   (bad)
```

This is an **experimental architectural pivot** inspired by robust LA caching,
selective prediction/abstention, and regret-driven trust adaptation. No theorem guarantee is claimed.

## Learned gating proof-of-concept

`ml_gate_v1` is intentionally experimental and lightweight: logistic regression over decision-time features.
It does **not** learn full eviction, only the gate between predictor and LRU experts.

## Learned gating v2 (counterfactual labels)

`ml_gate_v2` evaluates predictor-victim vs LRU-victim choices by bounded-horizon local counterfactual replay, producing a regression regret target (`y_reg`) and derived binary gate target (`y_cls`).

## Direct eviction-value pivot (evict_value_v1)

`evict_value_v1` is an **experimental structural pivot** away from trust-gating and calibration-heavy blending.
At each full-cache miss, it scores every cached candidate with a regression model that predicts:

```text
y_loss(q,t;H) = misses over next H requests
                after forcing eviction of candidate q at time t
                and replaying with LRU transitions
```

The online policy evicts `argmin_q predicted_loss(q,t;H)`.

This is candidate-centric supervision (direct eviction quality), not a gate over predictor-vs-LRU experts.
