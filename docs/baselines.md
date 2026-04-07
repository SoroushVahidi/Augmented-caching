# Baselines: Learning-Augmented Weighted Paging

## Paper Citation

> Bansal, N., Coester, C., Kumar, R., Purohit, M., & Vee, E. (2022).
> **Learning-Augmented Weighted Paging.**
> *Proceedings of the 33rd Annual ACM-SIAM Symposium on Discrete Algorithms (SODA 2022).*

---

## Problem Setting

We implement **Baseline 1**: the deterministic learning-augmented weighted
paging algorithm from the paper above.

### Formal setting

- A cache of capacity **k** (number of pages; all pages have unit size).
- A finite universe of pages.  Each page `p` has a fetch cost **w_p > 0**.
- A request sequence `σ_1, σ_2, ..., σ_T`.
- At time `t`, the algorithm observes `σ_t` **and** a prediction `τ_t`:
  the predicted next time index at which `σ_t` will be requested again.
  `τ_t = ∞` means "predicted to never be requested again."
- On a cache miss for `σ_t`, the algorithm pays `w_{σ_t}` and fetches the page.
- **Objective:** minimise total fetch cost.

---

## What Was Implemented

| Module | Description |
|--------|-------------|
| `src/lafc/types.py` | `Page`, `Request`, `CacheEvent`, `SimulationResult` dataclasses |
| `src/lafc/simulator/request_trace.py` | JSON trace loader; fills `actual_next` by forward scan |
| `src/lafc/simulator/cache_state.py` | `CacheState`: add/evict/query with capacity enforcement |
| `src/lafc/policies/base.py` | `BasePolicy` abstract base class |
| `src/lafc/policies/lru.py` | Least-Recently-Used (LRU) baseline |
| `src/lafc/policies/weighted_lru.py` | Weighted-LRU: evict minimum-weight page on fault |
| `src/lafc/policies/advice_trusting.py` | Belady-with-predictions: evict `argmax τ_q` |
| `src/lafc/policies/la_weighted_paging_deterministic.py` | **Core algorithm** (see below) |
| `src/lafc/policies/la_weighted_paging_randomized.py` | Scaffold only (see below) |
| `src/lafc/predictors/offline_from_trace.py` | Perfect oracle: `τ_t := a_t` |
| `src/lafc/predictors/noisy.py` | Additive noise, random class swaps, bounded inversions |
| `src/lafc/metrics/cost.py` | `total_fetch_cost`, `hit_rate`, `per_page_cost` |
| `src/lafc/metrics/prediction_error.py` | η metric + per-class weighted surprises |
| `src/lafc/runner/run_policy.py` | CLI runner; outputs `summary.json`, `metrics.json`, `per_step_decisions.csv` |

---

## Deterministic Algorithm (`la_det`)

### Core idea

The algorithm is a **prediction-guided, weight-normalised eviction** policy.
On a cache miss for page `p`:

1. Pay `w_p`.
2. For each currently cached page `q ≠ p`, compute

       eviction_score(q) = τ_q / w_q

   where `τ_q` is the most recently received predicted next-arrival of `q`.

3. Evict `q* = argmax eviction_score(q)`.

4. Fetch `p` into cache.

Additionally, on every request (hit or miss), the prediction `τ_t` is stored
for `σ_t` and used in future eviction decisions.

### Pseudocode

```
State:  predicted_next[p]  for all p   (initialised to ∞)
        cache              ⊆ pages     (initialised empty)

On request t for page p with prediction τ_t:
    predicted_next[p] ← τ_t
    if p ∈ cache:
        return HIT
    else:
        pay w_p
        if |cache| = k:
            q* ← argmax_{q ∈ cache} { predicted_next[q] / w_q }
            cache ← cache \ {q*}
        cache ← cache ∪ {p}
        return MISS, evicted=q*
```

### Paper-to-code mapping

| Concept | Code location |
|---------|--------------|
| Weight classes | `WeightClass` dataclass; `_weight_classes` dict in `LAWeightedPagingDeterministic` |
| Prediction storage | `_predicted_next: Dict[PageId, float]` |
| Eviction score | `_eviction_score()` method |
| Eviction candidate selection | `_choose_eviction_candidate()` method |
| Fetch-cost payment | `_record_miss(cost)` in `on_request()` |

---

## Exact Assumptions

1. All pages have unit size (only fetch cost differs).
2. Predictions `τ_t` are for the **next arrival time** of `σ_t` after time `t`.
3. The algorithm may not look ahead in the trace (online).
4. `actual_next` is computed from the full trace offline and is used only for
   evaluation metrics, **never** for eviction decisions.

---

## Interpretation Notes

### INTERPRETATION NOTE 1 — Eviction score formula

The paper's deterministic algorithm (Theorem 1) is described in terms of a
primal-dual / water-filling mechanism.  The precise per-step update equations
require careful reading of the full paper and appendix.

Our implementation uses `eviction_score(q) = τ_q / w_q` as the eviction
ordering.  This is motivated as follows:

- **Within a single weight class** (all `w_q` equal), this reduces to
  `τ_q / constant = k · τ_q`, which is exactly **Belady-with-predictions**:
  evict the page predicted to be needed farthest in the future.
  This is O(1)-consistent (matches OPT when predictions are perfect).

- **Across weight classes**, dividing by `w_q` means that cheaper pages are
  evicted more readily for the same predicted next-arrival value.  This
  matches the paper's intuition that eviction rate from class `i` should be
  proportional to `1/w_i`.

We believe this interpretation is consistent with the paper's structural
guarantees but cannot be verified without access to the full algorithm
statement.  Any deviation from the true algorithm is an approximation.

### INTERPRETATION NOTE 2 — Default predicted_next

When the algorithm has not yet received a prediction for a page, we
initialise `predicted_next[p] = ∞`.  This means the page is treated as
"predicted to never be needed again" and is therefore a candidate for
eviction.

An alternative default is `0` (needed immediately), which would make the
page harder to evict.  We chose `∞` as the more conservative option that
matches the typical "don't evict pages you know are needed soon" intuition
— a page with unknown prediction is treated like one predicted to stay away
a long time.

### INTERPRETATION NOTE 3 — Weighted surprises metric

The paper's ε-like weighted surprise metric is based on inversions in the
prediction ordering within each weight class.  Our implementation counts
disagreements (`τ_t ≠ a_t`) per class, which is an upper bound on the true
inversion count.  A full inversion-pair count would require O(T²) comparisons
per class and is deferred to future work.

---

## Deviations from the Paper

1. The exact water-filling / primal-dual update equations are not implemented;
   the eviction score formula is an interpretation (see Interpretation Note 1).
2. The Belady oracle is not implemented separately; `advice_trusting` serves
   as the prediction-perfect baseline.

---

## Theorem-Style Properties Not Verified in Code

- **Theorem 1 (Deterministic, SODA 2022):** The deterministic algorithm is
  O(1)-consistent and O(log k)-robust.  These competitive-ratio bounds are
  not verified empirically or formally in the test suite.

- **Theorem 2 (Randomized, SODA 2022):** The randomized algorithm achieves
  O(log ℓ)-robustness where ℓ = number of weight classes.  Not implemented.

---

## Randomized Algorithm Status

**SCAFFOLDED — NOT IMPLEMENTED.**

See `src/lafc/policies/la_weighted_paging_randomized.py` for:
- Why implementation was deferred.
- A full list of TODO markers tied to specific paper sections.

The randomized algorithm requires:
1. A hierarchical decomposition of weight classes.
2. A phase-structured fractional water-filling update.
3. A randomized rounding / coupling step.

These are individually well-defined but require a careful reading of the
full proof to avoid misrepresenting the paper's contribution.

---

## How to Run the Baseline

### Install

```bash
pip install -e ".[dev]"
```

### Smoke test

```bash
python -m lafc.runner.run_policy \
    --policy   la_det \
    --trace    data/example.json \
    --capacity 3
```

### Toy experiment: compare all baselines

```bash
for policy in lru weighted_lru advice_trusting la_det; do
    echo "=== $policy ==="; \
    python -m lafc.runner.run_policy \
        --policy   $policy \
        --trace    data/example.json \
        --capacity 3; \
done
```

### With perfect predictions

```bash
python -m lafc.runner.run_policy \
    --policy la_det \
    --trace  data/example.json \
    --capacity 3 \
    --perfect-predictions
```

### Run tests

```bash
pytest tests/ -v
```

---

## Input Format

JSON file with three fields:

```json
{
    "requests":    ["A", "B", "C", "A", "B"],
    "weights":     {"A": 1.0, "B": 2.0, "C": 4.0},
    "predictions": [3, 4, 9999, 9999, 9999]
}
```

- `requests` (required): ordered list of page identifiers (strings or ints).
- `weights` (required): mapping from page id to fetch cost (must be > 0).
- `predictions` (optional): list of predicted next-arrival times, aligned
  with `requests`.  Omit to use `math.inf` (all unknown), or use
  `--perfect-predictions` to generate from the trace.

---

## Output Files

All written to `--output-dir` (default: `output/`):

| File | Contents |
|------|----------|
| `summary.json` | `policy_name`, `total_cost`, `total_hits`, `total_misses`, `hit_rate` |
| `metrics.json` | `prediction_error_eta`, `total_weighted_surprise`, `per_class_surprises` |
| `per_step_decisions.csv` | `t`, `page_id`, `hit`, `cost`, `evicted` — one row per request |
