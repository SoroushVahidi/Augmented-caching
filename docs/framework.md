# Experimental Framework: Confidence-Aware Learning-Augmented Caching

## Status

This framework is **experimental** and currently implements a **first-version policy** (`atlas_v1`).
It is intended for empirical comparisons against existing baselines in this repository.

**Important:** this implementation does **not** claim a proved theorem or competitive guarantee.

## Goal

At each eviction, combine:
1. a prediction-derived score,
2. a confidence-aware trust weight,
3. a robust fallback score from LRU.

## ATLAS v1 (what is implemented)

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
