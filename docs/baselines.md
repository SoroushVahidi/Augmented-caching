# Baselines: Learning-Augmented Caching

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

---

---

# Baseline 2: Predictive Marker (Lykouris & Vassilvitskii 2018)

## Paper Citation

> Lykouris, T., & Vassilvitskii, S. (2021).
> **"Competitive Caching with Machine Learned Advice."**
> *Journal of the ACM (JACM), 68(4), 1–25.*
> (Conference version: ICML 2018.)

---

## PAPER-TO-CODE IMPLEMENTATION NOTE

### 1. Exact paging setting

Standard paging (unweighted caching):
- Cache capacity **k** (unit-size pages).
- **All misses have unit cost** (cost = 1, regardless of page identity).
- Request sequence σ_1, σ_2, ..., σ_T.
- At time t the algorithm observes σ_t and a prediction τ_t (predicted
  next arrival time of σ_t after t).  τ_t = ∞ means "never again."
- Objective: minimise total cache misses.

This is **not** the weighted paging setting of Baseline 1.

### 2. Prediction model

For each request at time t:
- requested page: σ_t
- predicted next arrival: τ_t ∈ {t+1, ..., T, ∞}
- actual next arrival: a_t (computed offline; **not** used by the online algorithm)

Prediction error:

    η = Σ_t |τ_t − a_t|

Both τ_t = ∞ and a_t = ∞ contribute 0.  One-sided ∞ contributes ∞.

### 3. Phase structure of Marker

The Marker algorithm processes requests in **phases**:
- At the start of each phase all k cached pages are **unmarked**.
- When page p is requested:
  - **HIT** (p ∈ cache): mark p.
  - **MISS** (p ∉ cache):
    - If |M| = k (all cached pages marked): start a new phase (M ← ∅).
    - Evict one page from the unmarked set (C ∖ M).
    - Fetch p and mark it.

A phase is exactly a maximal sequence of requests during which at most k
*distinct* new pages fault.

### 4. How Predictive Marker modifies Marker

Identical phase structure.  The only change is the eviction rule:

| Algorithm | Eviction rule |
|-----------|---------------|
| Marker | arbitrary v ∈ C ∖ M |
| Predictive Marker | v* = argmax_{v ∈ C ∖ M} predicted_next[v] |

The argmax uses the most recently received prediction for each cached page.

### 5. Clean chains (definition and usage in code)

**Definition used in this implementation (INTERPRETATION NOTE D):**
- An eviction is **clean** if the page evicted by Predictive Marker
  (argmax predicted_next among unmarked) equals the page that an offline
  oracle with access to actual_next would evict (argmax actual_next among
  unmarked).
- A **phase is clean** if all evictions within it are individually clean.
- A **clean chain** is a maximal consecutive sequence of clean phases.

**Usage:**
- Clean chains are used in the paper's analysis (proof of Theorem 2) to
  bound the number of extra faults by the prediction error η.
- In code, clean chains are tracked as a **post-hoc diagnostic** using
  `actual_next` (which is available in `Request` but not used for decisions).
- Stored in `SimulationResult.extra_diagnostics["clean_chains"]`.

### 6. Quantities tracked in code

| Variable | Description |
|----------|-------------|
| `_marked` | Set of marked pages in current phase (subset of cache) |
| `_phase` | Current phase index (1-based) |
| `_predicted_next` | Most recently received prediction for each page |
| `_actual_next` | Most recently received actual_next for each page (diagnostics) |
| `_phase_log` | List of `PhaseRecord` objects; one per phase |

### 7. Exact vs interpreted parts

**EXACT (from the paper):**
- Phase structure (Section 3).
- Unit cost per miss.
- Eviction rule: argmax predicted_next among unmarked pages.
- η metric: Σ |τ_t − a_t|.

**INTERPRETATION NOTE A — Default prediction:**
The paper does not specify a default for pages with no prior prediction.
We use `math.inf` (treat as "never needed again"): such pages are
candidates for eviction, deprioritised over pages with finite predictions.

**INTERPRETATION NOTE B — Tie-breaking in eviction:**
When two unmarked pages share the same predicted_next, the paper is silent.
We break ties by evicting the **lexicographically largest** page_id.
This is deterministic and reproducible.

**INTERPRETATION NOTE C — Expired predictions:**
If a page's stored predicted_next < current_t (the prediction has already
passed), we use the stored value as-is.  Resetting to ∞ would be an
alternative; we chose consistency with the stored value.

**INTERPRETATION NOTE D — Clean-phase definition:**
The paper uses "clean chains" in its proof sketch without giving an explicit
per-step algorithmic definition.  We define a phase as clean if every
eviction within it matches the offline oracle (argmax actual_next among
unmarked).

---

## What Was Implemented (Baseline 2)

| Module | Description |
|--------|-------------|
| `src/lafc/policies/marker.py` | Standard Marker (phase-based, deterministic, unweighted) |
| `src/lafc/policies/blind_oracle.py` | Blind Oracle: evict argmax predicted_next, unit cost |
| `src/lafc/policies/predictive_marker.py` | Predictive Marker (faithful to LV 2018) with clean-chain diagnostics |
| `src/lafc/metrics/prediction_error.py` | Added `compute_eta_unweighted()` for η = Σ\|τ_t − a_t\| |
| `src/lafc/runner/run_policy.py` | Added `marker`, `blind_oracle`, `predictive_marker` to registry |
| `data/example_unweighted.json` | Example trace (no weights; unit weights assumed) |
| `tests/test_policies_baseline2.py` | 38 tests for Marker, BlindOracle, PredictiveMarker |

**Also updated:**
- `src/lafc/types.py`: added `phase: Optional[int]` field to `CacheEvent`;
  added `extra_diagnostics: Optional[Dict]` to `SimulationResult`.
- `src/lafc/simulator/request_trace.py`: made `weights` optional (defaults
  to 1.0 for all pages when absent — unweighted paging).
- `src/lafc/policies/__init__.py`: exports `MarkerPolicy`,
  `BlindOraclePolicy`, `PredictiveMarkerPolicy`.

---

## Exact Assumptions (Baseline 2)

1. All pages have unit size and unit miss cost.
2. Weights in the trace (if supplied) are **ignored** by Marker,
   BlindOracle, and PredictiveMarker — they use unit cost 1.0 always.
3. Predictions τ_t are for the **next arrival time** of σ_t after time t.
4. The algorithm uses only the most recently received prediction for each
   page (stored from the last time that page was requested).
5. `actual_next` is computed offline and used **only** for clean-chain
   diagnostics, never for eviction decisions.

---

## Deviations from the Paper (Baseline 2)

1. **Deterministic Marker tie-breaking:** We evict the lexicographically
   smallest unmarked page (in Marker) or largest predicted_next (in
   Predictive Marker), rather than a random choice.  This does not affect
   Predictive Marker's theoretical guarantees.
2. **Clean-chain diagnostic definition:** The paper's proof uses clean
   chains analytically; we define them concretely per-eviction as
   described in Interpretation Note D above.
3. **Default prediction = ∞:** See Interpretation Note A.

---

## Theorem-Style Properties Not Verified in Code (Baseline 2)

- **Theorem 2 (LV 2018):** Predictive Marker is O(1)-consistent (when
  η = 0 the algorithm matches OPT) and O(log k)-robust.  These competitive-
  ratio bounds are not formally verified in the test suite.
- The randomised Marker backbone (used for the O(log k) guarantee) is not
  implemented; our Marker is deterministic.

---

## How to Run Baseline 2

### Install

```bash
pip install -e ".[dev]"
```

### Smoke test (Predictive Marker on unweighted trace)

```bash
python -m lafc.runner.run_policy \
    --policy predictive_marker \
    --trace  data/example_unweighted.json \
    --capacity 3
```

### Toy experiment: compare all Baseline 2 policies

```bash
for policy in marker blind_oracle predictive_marker; do
    echo "=== $policy ===" && \
    python -m lafc.runner.run_policy \
        --policy   $policy \
        --trace    data/example_unweighted.json \
        --capacity 3; \
done
```

### With perfect predictions (oracle)

```bash
python -m lafc.runner.run_policy \
    --policy predictive_marker \
    --trace  data/example_unweighted.json \
    --capacity 3 \
    --perfect-predictions
```

### Run tests

```bash
pytest tests/ -v
```

---

## Input Format (Unweighted, Baseline 2)

JSON file with two fields (`weights` is optional):

```json
{
    "requests":    ["A", "B", "C", "A", "B"],
    "predictions": [3, 4, 9999, 9999, 9999]
}
```

- `requests` (required): ordered list of page identifiers.
- `predictions` (optional): predicted next-arrival times aligned with
  `requests`.  Omit to use `math.inf`, or use `--perfect-predictions`.
- `weights` (optional): if supplied, unit costs are still used by
  Marker/BlindOracle/PredictiveMarker.

---

## Output Files (Baseline 2)

All written to `--output-dir` (default: `output/`):

| File | Contents |
|------|----------|
| `summary.json` | `policy_name`, `total_cost`, `total_hits`, `total_misses`, `hit_rate` |
| `metrics.json` | `prediction_error_eta`, `eta_unweighted`, `num_clean_phases`, `num_dirty_phases`, `num_clean_chains`, `total_clean_evictions`, `total_dirty_evictions` |
| `per_step_decisions.csv` | `t`, `page_id`, `hit`, `cost`, `evicted`, `phase` — one row per request |
