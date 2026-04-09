# Baselines: Learning-Augmented Caching

## Baseline 1: Learning-Augmented Weighted Paging (SODA 2022)

### Paper citation

Bansal, N., Coester, C., Kumar, P., Purohit, M., & Vee, E. (2022).
**Learning-Augmented Weighted Paging**. SODA 2022.

### Policy names and status

- `la_det` / `la_det_approx`:
  historical interpreted heuristic (`predicted_next / weight` style ranking).
  Kept for backward compatibility and ablation, **not** paper-faithful.
- `la_det_faithful`:
  class-level deterministic process implementation attempt using explicit
  `x_i`, `mu_i`, and interval-set `S_i` state with continuous dynamics
  discretized in small pointer steps.

### Discretization details for `la_det_faithful`

1. Classes are grouped by exact weights.
2. Predictions are used only for within-class rank ordering.
3. Pointer `p` moves from `q-1` to `q`; dynamics activate only when `p > x_r`.
4. ODE terms for `x_i'` and `mu_i'` are integrated with fixed step Euler updates.
5. `S_i` is represented as a finite union of intervals with union/remove/clip ops.
6. Dummy-page boundary handling is implemented via internal mass floor `x_i >= 1`
   and total effective mass `sum_i x_i = k + ell` (real cache still size `k`).

### Caveat

`la_det_faithful` is a faithful-style numerical reimplementation attempt of the
paper's continuous class-level process. It is substantially closer to the paper
than `la_det`, but still a discretized simulation rather than an exact symbolic
continuous solver.

---

## Baseline 3 (Main): TRUST&DOUBT

### Paper citation

Antoniadis, A., Coester, C., Eliáš, M., Polak, A., & Simon, B. (2020). **Online Metric Algorithms with Untrusted Predictions**. ICML 2020, PMLR 119:345–355. Supplementary material contains Algorithm 3 pseudocode.

## Step-0 paper-to-code note (TRUST&DOUBT)

1. **Setting implemented**: unweighted paging (unit miss cost), cache size `k`, request sequence `r_t`.
2. **Prediction role**: TRUST&DOUBT uses **predicted cache states** `P_t` (predictor configuration at time `t`), not directly next-arrival values.
3. **State maintained**: `A` (ancient), `stale`, `U`, `M`, `C`, `T`, `D`, and for each clean page `q`: `p_q`, `trusted(q)`, threshold `t_q`, and `q_interval_change`.
4. **Eviction/update logic**: implemented from Supplementary Algorithm 3 (phase reset; steps 1–4; threshold doubling across doubted intervals).
5. **Difference vs Blind Oracle / Predictive Marker**:
   - Blind Oracle fully trusts predicted next-arrival advice.
   - Predictive Marker remains a marking algorithm.
   - TRUST&DOUBT adaptively alternates trust/doubt and may evict marked pages (through set `T`).
6. **Interpretation-required points**:
   - Paper says “arbitrary” choices in several places; implementation now uses seeded randomness (`--trust-seed`) for those choices.
   - Paper describes non-lazy formulation; implementation follows Remark 10 by simulating non-lazy cache in background and serving requests lazily.
   - Caching distance for MTS-style error is interpreted as `|X \ Y|` between equal-size cache states.
7. **Mapping to paper sections**:
   - Main description: ICML paper Section 4.
   - Full operational pseudocode: Supplementary Algorithm 3.
   - Non-lazy/lazy implementation note: Remark 10.
8. **Faithfulness status**:
   - Previous implementation was partly interpreted and overly deterministic in arbitrary branches.
   - Current implementation keeps explicit algorithmic state (`A, stale, C, U, M, T, D`, plus `p_q`, `trusted(q)`, `t_q`, interval boundaries) and uses seeded randomized arbitrary choices.
9. **Predictor cache representation**:
   - Native interface: `Request.metadata["predicted_cache"]` as a per-step list/set of page ids.
   - JSON traces may include top-level `predicted_caches`; CSV traces may include `predicted_cache` as `A|B|...`.
   - `--derive-predicted-caches` remains an adapter that converts next-arrival predictions to cache-state predictions via Blind-Oracle conversion (Sec. 1.3), not the native paper interface.

---

## Baseline 4: Deterministic BlindOracle + LRU combiner (Wei 2020)

### Paper citation

Wei, A. (2020). **Better and Simpler Learning-Augmented Online Caching**. APPROX/RANDOM 2020.

### Step-0 paper-to-code note (Baseline 4)

1. **State maintained**:
   - shadow BlindOracle policy state,
   - shadow LRU policy state,
   - cumulative miss counts for each shadow,
   - deterministic tie-breaking state (`BlindOracle` on ties).
2. **BlindOracle definition**: unweighted paging, unit miss cost, on miss+full cache evict cached page with maximum predicted next-arrival; deterministic tie-breaking.
3. **LRU definition**: deterministic standard unweighted LRU.
4. **Cost tracking**: both shadows process every request; misses are accumulated online.
5. **Operational meaning of “follow whichever performed better so far”**:
   - At time `t`, compare shadow miss counts from requests `0..t-1`.
   - If BO misses <= LRU misses, follow BO at `t`; else follow LRU.
   - The combiner’s event for request `t` is the selected shadow’s event.
6. **Tie-breaking**: ties go to BlindOracle.
7. **Faithfulness status**:
   - Previous implementation used a third independent cache and applied the leading algorithm’s *rule* to that cache.
   - Current implementation follows the selected shadow algorithm directly (literal follow-leader interpretation).

### Caveats / interpretation notes

- The paper references black-box combiner machinery (Fiat et al. / Blum-Burch) and gives a concise informal description.
- Where lower-level operational details are not specified, this repository uses deterministic tie-breaking and explicit shadow-following semantics documented in code as `INTERPRETATION NOTE`.

---

## Baseline 4b: RobustFtP-D with MARKER fallback (Chłędowski et al., 2021)

### Paper citation

Chłędowski, J., Polak, A., Szabucki, B., & Żołna, K. (2021).
**Robust Learning-Augmented Caching: An Experimental Study**. ICML 2021, PMLR 139.

### Implemented policy names

- `robust_ftp_d_marker` (primary)
- `robust_ftp` (alias)

### Why this is separate from `blind_oracle_lru_combiner`

The existing `blind_oracle_lru_combiner` is a Wei-2020-style interpreted
combiner over **reuse-distance** predictor advice (BlindOracle expert + LRU).
The ICML'21 RobustFtP baseline instead uses **policy predictions** and combines:

1. a robust fallback policy (MARKER in the paper's main reported variants), and
2. a predictor-following policy expert.

So this repository now keeps both baselines explicitly to avoid confusion.

### Exact implemented variant

- Deterministic RobustFtP variant (RobustFtPD spirit) with MARKER fallback.
- Combiner rule: follow the lower cumulative-miss expert so far (deterministic
  tie-break to predictor expert), logging switch points and expert trajectories.
- Predictor expert uses `metadata["predicted_cache"]` and on full-cache misses
  evicts a page in `cache \\ predicted_cache` (deterministic tie-break).

### Faithfulness assessment

- **Mostly faithful with minor interpretation.**
- High-level structure (robust expert + predictor-following expert + deterministic
  switching) is paper-aligned.
- Low-level tie-breaking and exact black-box combiner internals are interpreted
  because the experiment section does not fully specify every implementation detail.

### Diagnostics exposed

- `robust_ftp_followed_predictor_steps`
- `robust_ftp_followed_robust_steps`
- `robust_ftp_switch_count`
- `robust_ftp_switch_fraction`
- `robust_ftp_shadow_predictor_total_misses`
- `robust_ftp_shadow_robust_total_misses`
- plus per-step decision log in `extra_diagnostics["robust_ftp"]["step_log"]`
  and switch timestamps in `extra_diagnostics["robust_ftp"]["switch_points"]`.

---

## Baseline 5: Parsimonious Learning-Augmented Caching (Im et al., 2022)

### Paper citation

Im, S., Kumar, R., Petety, A., & Purohit, M. (2022).
**Parsimonious Learning-Augmented Caching**. ICML 2022, PMLR 162.

### Implemented policy names

- `adaptive_query` (primary paper-faithful naming)
- `parsimonious_caching` (CLI alias)

### Exact implemented variant

- Implemented algorithm: **AdaptiveQuery-b** (paper Section 5, Algorithm 3),
  with the Section 5.3 robust modification used in Theorem 11:
  switch to random unmarked eviction when chain depth exceeds `log k`.
- Query budget behavior: on query-mode misses, sample up to `b` unmarked
  pages uniformly at random without replacement, query only those pages, and
  evict the sampled page with largest predicted next-arrival.
- Randomness is seedable via CLI (`--adaptive-query-seed`).

### Diagnostics exposed

- `adaptive_query_queries_used`
- `adaptive_query_fraction_misses_queried`
- `adaptive_query_fraction_misses_fallback_random`
- `adaptive_query_avg_queries_per_queried_miss`
- `adaptive_query_query_mode_evictions`
- `adaptive_query_random_mode_evictions`
- `adaptive_query_max_chain_depth_seen`
- plus parameter echoes (`adaptive_query_b`, `adaptive_query_seed`)

### Interface adaptation note (important)

The paper's query model assumes the algorithm can call `Q(p,t)` for any cached
page at eviction time. This repository's standard trace interface provides
`predicted_next` only for the currently requested page. To stay consistent with
existing baselines and runner interfaces, `adaptive_query` interprets `Q(p,t)`
as the most recently observed prediction for page `p` (default `∞` if unseen).
This is a conservative adapter; it is documented explicitly in policy comments.

---

## Implemented baselines

- `lru`
- `marker`
- `blind_oracle`
- `predictive_marker`
- `trust_and_doubt` (Baseline 3 target)
- `blind_oracle_lru_combiner` (Baseline 4 target)
- `robust_ftp_d_marker` / `robust_ftp` (ICML'21 experimental robust switching)
- `adaptive_query` / `parsimonious_caching` (Baseline 5 target)
- `evict_value_v1_guarded` (experimental guard-style robust wrapper over `evict_value_v1`)

## Prediction interfaces supported

- **Next-arrival predictions** (`predictions` in trace).
- **Predicted cache states** (`predicted_caches` in trace / `metadata["predicted_cache"]`).
- Conversion utility from next-arrival predictions to `P_t` using Blind-Oracle conversion (paper Sec. 1.3).

## Error metrics exposed

- `prediction_error_eta` (legacy next-arrival eta).
- `eta_unweighted` (LV-style unweighted next-arrival eta).
- `cache_state_error_total` (MTS-style state error over `P_t` vs offline Belady states).

## Deviations / non-goals

- We did **not** implement full generic MTS algorithms; only paging-specialized machinery needed for baseline quality.
- We did **not** formally verify theorem constants/competitive-ratio proofs in code.

## Running

```bash
python -m lafc.runner.run_policy \
  --policy trust_and_doubt \
  --trace data/example_unweighted.json \
  --capacity 3 \
  --perfect-predictions \
  --derive-predicted-caches
```

Outputs:
- `summary.json`
- `metrics.json`
- `per_step_decisions.csv`
