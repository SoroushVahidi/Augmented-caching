# Practical-Significance Ablation Protocol (`practical_significance_ablation_v1`)

Reviewer 1, Major Comment 4: the current `evict_value_v1` implementation is
substantially slower per eviction decision than lightweight baselines while
not outperforming them end-to-end. This protocol answers *where the cost
comes from*, *how much of it is avoidable implementation overhead*, *whether
selective/pruned invocation recovers most of the quality cheaply*, and *at
what miss-cost regime (if any) the added cost would be justified* — without
touching any paid/external API.

Machine-readable companion: `configs/practical_significance_ablation_v1.json`.
Frozen before measurement; any change after seeing performance numbers
requires a `_v2`.

## 1. Existing overhead evidence (do not duplicate)

The manuscript's 75.0 / 152.1 / 316.0 ms per-eviction-decision numbers at
capacities 32/64/128 (Section `subsec:overhead_scalability`) come directly
from `scripts/run_overhead_benchmark.py`, whose output lives at
`analysis/kbs_overhead_benchmark_local_tmux_20260621.{csv,md}`. That script
times the *whole* `policy.on_request(request)` call with
`time.perf_counter()`, on a single trace (BrightKite), a 5,000-request
prefix, one run, no component breakdown. It is not duplicated here — this
protocol extends it (all 7 certified traces, repetitions, component
breakdown) rather than replacing it.

## 2. Where the cost actually comes from (source-level audit)

Read directly from `src/lafc/policies/evict_value_v1.py::_choose_victim` and
`src/lafc/evict_value_features_v1.py::compute_candidate_features_v1`:

- **Redundant O(k²) work.** `_choose_victim` loops over all `k` resident
  candidates and, for *each one*, calls `_build_candidate_features`, which
  internally calls `compute_predictor_scores(candidates, bucket_by_page)`
  and `compute_lru_scores(candidates)`. Both of those return a dict keyed by
  *every* candidate and do not depend on which specific candidate is being
  scored — so the canonical implementation recomputes the same O(k log k)
  work `k` times per decision (O(k² log k) total) instead of once
  (O(k log k)). This is a real, provable, decision-preserving optimization
  opportunity (`cached_exact`).
- **Redundant O(k · history_window) work.** `_build_candidate_features`
  computes `recent_candidate_request_rate`/`recent_candidate_hit_rate` by
  scanning `self._recent_req_hist`/`self._recent_hit_hist` (each up to
  `history_window` long) *per candidate*, when a single `Counter` pass over
  the history once per decision would give every candidate's rate in one
  pass.
- **Unbatched model inference.** `_choose_victim` calls
  `EvictValueV1Model.predict_loss_one` once per candidate, each of which
  wraps a length-1 `estimator.predict(x)` call. `EvictValueV1Model` already
  ships a `predict_loss_batch` method (`src/lafc/evict_value_model_v1.py`)
  that takes one batched `estimator.predict(x)` call over all `k` rows —
  it already exists in the codebase but is not used by the canonical
  policy's inner loop. This matters most for estimators with per-call fixed
  overhead (e.g. `HistGradientBoostingRegressor`'s internal thread-pool
  spin-up).
- **Per-candidate dict allocation.** `CandidateFeatureContext.as_dict()`
  allocates a fresh 26-key Python dict per candidate; this is Python/object
  overhead unrelated to the algorithm itself.

None of the above changes the *algorithm* — candidate-level finite-horizon
eviction-loss scoring is still fundamentally O(k) model evaluations per
decision (a framework-level cost, not avoidable without changing the
method). What the audit above identifies is **avoidable
research-prototype implementation overhead layered on top of that O(k) floor**
— this is exactly the distinction Section 4 of the reviewer task brief asks
the manuscript to be able to draw, and only measurement (not this source
reading) can confirm how large that avoidable share actually is.

## 3. Optimization variants (exact, decision-preserving)

Implemented in `src/lafc/policies/evict_value_v1_optimized.py`:

| Variant | What changes | Exact? |
|---|---|---|
| `canonical` | unmodified `EvictValueV1Policy` | — |
| `cached_exact` | hoists predictor/LRU scores, victim IDs, cache-level aggregate stats, and request/hit-rate counts out of the per-candidate loop | yes |
| `vectorized_exact` | canonical per-candidate feature construction, one batched `predict_loss_batch` call instead of `k` `predict_loss_one` calls | yes |
| `vectorized_cached_exact` | both of the above | yes |

"Exact" means: same trace, same model artifact, same capacity, same seed →
byte-identical hit/miss sequence and byte-identical eviction (victim)
sequence as canonical. This is enforced by
`tests/test_practical_significance_ablation.py`, not merely asserted.

## 4. Selective / hybrid invocation (quality-cost tradeoff, not exact)

Two simple, predeclared rules — deliberately not tuned on held-out data:

- **`disagreement_lru_sieve`**: invoke the learned scorer only when a cheap
  LRU victim and a cheap shadow-SIEVE victim disagree; otherwise evict the
  LRU victim directly. The shadow-SIEVE state (visited bits + hand pointer)
  is maintained over the *real* resident candidate set on every request
  regardless of whether the learned scorer is invoked that decision, so the
  disagreement signal never requires the expensive computation itself.
- **`periodic`**: invoke the learned scorer on 1 of every `K=4` evictions
  (fixed, predeclared); otherwise evict the LRU victim.

Two rules considered and explicitly **not** implemented in v1:
`candidate_ambiguity` (would need a predeclared cheap ambiguity signal
beyond this protocol's current scope) and `score_margin_confidence`
(rejected — computing a score margin already requires paying the full
learned-inference cost, which would make "selective" circular, per the task
brief's explicit warning).

## 5. Top-k candidate pruning

Cheap prefilter = LRU recency order (already tracked internally); the
learned scorer evaluates only the `k` oldest (most LRU-eligible) candidates.
Grid `k ∈ {4, 8, 16, 32}`, restricted per capacity to `k < capacity` (larger
k is not meaningful pruning). `k` is never chosen post-hoc from the
headline result — the full grid is always reported.

## 6. Model-complexity variants

Reuses the already-frozen h4 training grid
(`models/evict_value_wulver_v1_h4_{ridge,random_forest,hist_gb}.pkl` in the
main repository, same target/features/training data) to separate "cost from
model choice" from "cost from the supervision framework." No new model
search is performed.

## 7. Break-even miss-cost analysis

`TotalCost_p(Cmiss) = DecisionComputeCost_p + Misses_p · Cmiss`, where
`DecisionComputeCost_p` only accrues on evictions (full-cache misses), never
on every request. Break-even against baseline `b`:

```
Cmiss* = (ComputeCost_learned - ComputeCost_b) / (Misses_b - Misses_learned)
```

defined only when `Misses_b > Misses_learned`. If the learned method has
`>=` misses than `b`, the protocol requires reporting explicitly that **no
positive miss penalty makes the current learned policy preferable to `b`
under this simple additive model** — this negative case must never be
silently omitted. Baselines: LRU, SIEVE, FIFO-Reinsertion (fixed), plus a
4th baseline chosen programmatically as the certified fair-window
lowest-mean-misses original implementable baseline (never hand-picked).

## 8. Miss-cost sweep and weighted-cost sensitivity

Miss-cost sweep: log-spaced synthetic grid `{1us,...,10s}` — explicitly not
a claim about any real system's actual miss cost.

Weighted-cost: audited first. The canonical trace loader
(`build_requests_from_lists`) assigns every `Page.weight = 1.0` uniformly
whenever no explicit weights are supplied, which is always true for the 7
certified `.jsonl` traces used throughout this repository's fair
comparisons. **No real heterogeneous per-object cost metadata exists** in
the pipeline actually used for evaluation. The weighted-cost analysis is
therefore a predeclared synthetic sensitivity analysis only (log-normal
per-page cost multiplier, seed 0), never presented as a real-world
measurement.

## 9. Quality-latency Pareto frontier

Computed over `{lru, sieve, fifo_reinsertion, evict_value_v1 canonical,
vectorized_cached_exact, both selective variants, all topk variants, all
model-complexity variants}`. Dominated learned-policy points are always
reported, never hidden.

## 10. Hardware-controlled timing gate

Component-level profiling and the exact-optimization/selective/top-k
*correctness* checks (decision sequences, invocation rates, candidate
retention) do not need an idle machine — they are deterministic given a
fixed trace/model/seed. Only **final reviewer-facing wall-clock timing
numbers** require a controlled, idle machine: no other Concern-1/2/3 job
(`evict_cross_family_resume`, `objective_ablation_pipeline`,
`distribution_shift_2h_continue`, or the legacy full-stream LRB process)
actively consuming CPU, fixed single-threaded BLAS env vars, multiple
repetitions with warm-up discarded, median + variation reported. If those
jobs are active, timing may still be *implemented and smoke-tested*, but
must be classified `TIMING_CAMPAIGN_PREPARED_BUT_DEFERRED` rather than
reported as final evidence.

## 11. API experiment decision gate (evaluated at the end of the v1 run)

No API is called anywhere in v1. An API-backed case study (real
heterogeneous inference/miss costs) is recommended only if the non-API
picture above leaves a specific, named gap — e.g. a practically relevant
cost regime that the synthetic sweep cannot represent — decided once, at
the end of the non-API work, never mid-run and never merely because API
credits are available.
