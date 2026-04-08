# Experimental Framework: Confidence-Aware Learning-Augmented Caching

## Status

This framework is **experimental** and currently implements three framework policies:
- `atlas_v1` (first version),
- `atlas_v2` (confidence-aware, dynamically trust-adaptive iteration),
- `atlas_v3` (confidence-aware local-trust iteration; first CCLT version).
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

- `atlas_v1`, `atlas_v2`, and `atlas_v3` are all **experimental**.
- `atlas_v2` is **confidence-aware** and **dynamically trust-adaptive**.
- `atlas_v3` is **confidence-aware**, **local-trust**, and context-adaptive.
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
