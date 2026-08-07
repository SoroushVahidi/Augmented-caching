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

## Baseline 6: LRB — Learning Relaxed Belady (Song et al., NSDI 2020)

### Paper citation

Song, Z., Berger, D. S., Li, K., & Lloyd, W. (2020). **Learning Relaxed
Belady for Content Distribution Network Caching**. NSDI 2020, pp. 529–544.

### Official code source and pinned commit

[`sunnyszy/lrb`](https://github.com/sunnyszy/lrb) (BSD-2-Clause), commit
`9e8b4423383c01c4528deb447f152f0437a37c3a` (fetched 2026-08-06). See
`docs/lrb_method_spec.md` for the full paper-and-code-grounded specification.

### Implemented policy names

- `lrb` (native, this repository's simulator; requires the optional
  `lightgbm` dependency: `pip install 'lafc[lrb]'`).

### Exact implemented variant

- Candidate-level online eviction: on a full-cache miss, sample
  `sample_rate` (default 64) cached objects uniformly at random and evict
  the one with the largest LightGBM-predicted `log1p(time-to-next-request)`.
- Delayed-label training: a sampled object's label matures only when
  re-requested, force-evicted for exceeding the sliding `memory_window`, or
  when its post-eviction "ghost" metadata times out of the window — never
  from ground-truth future information.
- GBDT (LightGBM) regressor, `num_iterations=32, num_leaves=32,
  learning_rate=0.1, feature_fraction=0.8, bagging_fraction=0.8,
  bagging_freq=5` — exact from the official code.
- Documented cold-start fallback (part of the official design, not an ad hoc
  addition): plain LRU eviction before any model has trained, or whenever
  the LRU-tail object's age exceeds `memory_window`.
- Deterministic tie-break by smallest `page_id` (the official code's
  `std::sort` is not guaranteed stable here; this repository picks one
  fixed rule for reproducibility).

### Unit-size specialization (important)

This repository's manuscript evaluation is standard **unweighted paging**
(unit miss cost, capacity measured in object slots: 32/64/128). The official
LRB targets variable-sized, byte-capacity CDN caches. Every object's size is
held at a constant 1 here; the size feature and the
`objective="object_miss_ratio"` score/size interaction are kept
structurally but are numerically inert under this specialization. This is
**"LRB under unit-size specialization,"** not a reproduction of the paper's
byte-cache CDN experiments — do not cite byte-miss-ratio results from this
implementation; only request-miss/miss-ratio results are meaningful here
(and are numerically identical to a byte-miss ratio under unit size, so
nothing is lost).

### Faithfulness assessment

Native, algorithmic-level-faithful port of the official simulator core
(not the ATS system-level prototype). Every design decision is classified
in `docs/lrb_method_spec.md` as exact-from-paper, exact-from-code, a
required adaptation, or an optional deviation. The two `memory_window`/
`batch_size` numeric defaults are the only genuinely adaptation-required
values (the official defaults are CDN-scale constants that never fire at
this repository's request-count scale); both are validation-tunable via CLI
flags and are tuned on a held-out validation prefix (never the test region)
in `scripts/experiments/run_lrb_external_baseline.py`, mirroring the
paper's own per-trace tuning protocol.

### Diagnostics exposed

- `lrb_sample_rate`, `lrb_memory_window`, `lrb_batch_size`,
  `lrb_max_n_past_timestamps`, `lrb_num_iterations`, `lrb_num_leaves`,
  `lrb_learning_rate`, `lrb_seed`, `lrb_objective_is_object_miss_ratio`
  (config echoes)
- `lrb_n_retrain`, `lrb_model_trained`
- `lrb_n_in_cache_meta`, `lrb_n_ghost_meta`, `lrb_n_pending_rows`
- `lrb_n_force_eviction`, `lrb_n_cold_start_evictions`,
  `lrb_n_age_forced_evictions`, `lrb_n_model_ranked_evictions`,
  `lrb_n_candidates_sampled_total`
- Per-step `CacheEvent.diagnostics["mode"]` ∈
  `{hit, direct_admit, cold_start_lru, age_forced_lru, model_ranked}`,
  plus `candidate_count`.

### Running

```bash
# Requires the optional lightgbm dependency:
pip install 'lafc[lrb]'

python -m lafc.runner.run_policy \
  --policy lrb \
  --trace data/example_unweighted.json \
  --capacity 3 \
  --lrb-sample-rate 4 --lrb-memory-window 20 --lrb-batch-size 8
```

Full external-baseline comparison across all 7 manuscript trace families and
capacities 32/64/128:

```bash
python scripts/experiments/run_lrb_external_baseline.py
```

Outputs write to `analysis/external_learned_baselines/lrb/` — canonical
`*_heavy_r1` artifacts are never touched.

---

## Baseline 7: 3L-Cache — Low Overhead and Precise Learning-based Eviction (Zhou et al., FAST 2025)

### Paper citation

Zhou, W., Niu, Z., Xiong, Y., Fang, J., & Wang, Q. (2025). **3L-Cache: Low Overhead and Precise Learning-based Eviction Policy for Caches**. FAST 2025, pp. 237–254.

### Official code source and pinned commit

[`optiq-lab/3L-Cache`](https://github.com/optiq-lab/3L-Cache) (GPL-3.0), commit
`134cd159b635cdab75419a4281bed1a330fef31f` (fetched 2026-08-06). See
`docs/three_l_cache_method_spec.md` for the full paper-and-code-grounded specification.

### Implemented policy names

- `three_l_cache` (native, this repository's simulator; requires the optional
  `lightgbm` dependency: `pip install 'lafc[lrb]'`).

### Exact implemented variant

- **Bidirectional sampling**: eviction candidates are drawn both from newly admitted objects ("sampling from the head", `_quick_demotion`) and from long-resident objects via a *persistent* scan pointer walking the LRU order once per lap ("sampling from the tail", `_tail_scan_round`).
- **Batched, heap-based eviction**: candidates are scored once per resampling round (not once per miss) and pushed onto a shared min-heap; a run of evictions is drawn from that heap before the next resampling round. Stale heap entries (objects re-requested or already evicted since being scored) are detected via a side `pred_map` and silently skipped.
- **Delayed-label training**: a sampled object's label matures only when re-requested, or when its post-eviction "ghost" metadata times out of the window. A *dynamic* window-exit label is constructed (a running empirical maximum wait time, frozen at each retrain), not LRB's fixed `2*window` constant.
- **Auto-tuning**: automatic online tuning of five parameters (`h_sw, f, x, Q`, and the closed-form `n`) from accumulated eviction-outcome statistics, exactly as specified in the paper's Table 2. Can be disabled (`auto_tune=False`) to freeze them at initial values (the paper's own "default value" ablation variant).
- **GBDT (LightGBM) regressor**: `num_iterations=16, num_leaves=32, learning_rate=0.1, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5` — exact from the official code.
- **Cold-start fallback**: plain LRU eviction before any model has trained (paper Section 4.2.1, footnote 3).
- **Deterministic tie-break** by smallest `page_id` for equal predicted values on the heap.

### Unit-size specialization (important)

This repository's manuscript evaluation is standard **unweighted paging** (unit miss cost, capacity measured in object slots: 32/64/128). The official 3L-Cache targets variable-sized, byte-capacity file/storage and CDN caches. Every object's size is held at a constant 1.0 here; the size feature and the `objective="byte_miss_ratio"` score interaction are kept structurally but are numerically specialized under this setting. This is **"3L-Cache under unit-size specialization,"** not a reproduction of the paper's byte-cache CDN/block-storage experiments — only request-miss/miss-ratio results are evaluated.

### Faithfulness assessment

Native, algorithmic-level-faithful port of the official simulator core. Every design decision is classified in `docs/three_l_cache_method_spec.md` as exact-from-paper, exact-from-code, a required adaptation, or an optional deviation. The `batch_size` numeric default is the only adaptation-required value (the official default is CDN-scale 64K which never fires at this repository's request-count scale); it is validation-tunable via CLI flags and is tuned on a held-out validation prefix (never the test region) in `scripts/experiments/run_three_l_cache_comparison.py`, mirroring the paper's own per-workload tuning protocol.

### Diagnostics exposed

- `three_l_cache_batch_size`, `three_l_cache_num_iterations`, `three_l_cache_num_leaves`,
  `three_l_cache_learning_rate`, `three_l_cache_seed`, `three_l_cache_objective_is_byte_miss_ratio`,
  `three_l_cache_auto_tune` (config echoes)
- `three_l_cache_n_retrain`, `three_l_cache_model_trained`
- `three_l_cache_n_in_cache_meta`, `three_l_cache_n_ghost_meta`, `three_l_cache_n_pending_rows`
- `three_l_cache_n_cold_start_evictions`, `three_l_cache_n_model_ranked_evictions`,
  `three_l_cache_n_stale_heap_pops`, `three_l_cache_n_resample_rounds`
- `three_l_cache_hsw`, `three_l_cache_f`, `three_l_cache_x`, `three_l_cache_q`
- Per-step `CacheEvent.diagnostics["mode"]` ∈ `{hit, direct_admit, cold_start_lru, model_ranked, heap_exhausted_lru_fallback}`.

### Running

```bash
# Requires the optional lightgbm dependency:
pip install 'lafc[lrb]'

python -m lafc.runner.run_policy   --policy three_l_cache   --trace data/example_unweighted.json   --capacity 3   --three-l-cache-batch-size 4 --three-l-cache-seed 0
```

Full external-baseline comparison across all 7 manuscript trace families and capacities 32/64/128:

```bash
python scripts/experiments/run_three_l_cache_comparison.py
```

Outputs write to `analysis/external_learned_baselines/three_l_cache/` — canonical `*_heavy_r1` artifacts are never touched.

---

## Baseline 8: HALP — Heuristic Aided Learned Preference Eviction Policy (Song et al., NSDI 2023)

### Paper citation

Song, Z., Chen, K., Sarda, N., Altınbüken, D., Brevdo, E., Coleman, J., Ju, X.,
Jurczyk, P., Schooler, R., & Gummadi, R. (2023). **HALP: Heuristic Aided
Learned Preference Eviction Policy for YouTube Content Delivery Network**.
20th USENIX Symposium on Networked Systems Design and Implementation
(NSDI '23).

### Official code source and pinned commit

**None.** HALP runs inside Google's proprietary YouTube CDN production
stack; no author-released simulator, reference implementation, or artifact
package is public (checked 2026-08-06: USENIX conference page, Google
Research publication page, author pages, general code-hosting search — see
`docs/halp_provenance.md`). This baseline is an independent
reimplementation of the paper's and the official Google Research blog
post's algorithmic description, not a port of any released code.

### Implemented policy names

- `halp` (native, this repository's simulator; no optional dependency
  beyond the core `scikit-learn` requirement).

### Exact implemented variant

- **Deterministic 8-candidate shortlist**: eviction candidates are the 8
  oldest pages currently in cache, read off the LRU tail. This
  deterministically approximates the paper's randomized, priority-queue-
  approximating baseline-heuristic sampling (see
  `docs/halp_method_spec.md`).
- **Pairwise preference target**: for two shortlisted candidates, the one
  re-accessed sooner (by ground-truth `actual_next`, observed only during
  the training window) is preferred.
- **Two-layer MLP reward model** (hand-rolled shared-weight
  `R(x) = W2 . relu(W1 x + b1)`, default 8 hidden units), trained once via
  deterministic full-batch gradient descent on a Bradley-Terry / RankNet
  pairwise cross-entropy loss — matching the official blog's "light-weight
  two-layer multilayer perceptron" characterization, with the
  online-vs-frozen-split and optimizer choice disclosed as adaptations.
  (Hand-rolled rather than `sklearn.neural_network.MLPClassifier` because a
  classifier trained on feature-difference vectors is only equivalent to a
  proper pairwise-scored network for a *linear* reward function; see
  `docs/halp_method_spec.md` and `docs/halp_provenance.md` §3.)
- **Cold-start fallback**: plain LRU eviction before the model is trained.
- **Frozen single-split training**: trained once at `training_trigger`
  (default 10,000 requests, i.e. the first 20% of a 50,000-request trace)
  and never retrained — an adaptation of the paper's continuous-online
  training regime to this offline-replay simulator.
- **Deterministic tie-break** by largest `page_id` among equally-scored
  candidates.

### Unit-size specialization (important)

Official HALP targets a variable-sized, byte-capacity production CDN DRAM
cache. Every object's size is held at 1.0 here, matching this repository's
canonical unweighted-paging evaluation (capacities 32/64/128 object slots).
This is **"HALP adapted to the unit-size, offline-replay paging setting,"**
not a reproduction of the paper's production byte-cache deployment —
only request-miss/miss-ratio results are evaluated, and no byte-miss-ratio
or CPU-overhead claim from the paper is reproduced or compared against.

### Faithfulness assessment

Independent reimplementation of the algorithmic core (heuristic-generated
shortlist → pairwise preference model → lowest-score eviction), verified
for **decision-semantic parity** (candidate/label/victim-selection
direction, leakage isolation) against the paper and official blog — not
model-architecture parity, loss-formulation parity, or production-score
parity, none of which is checkable without access to the closed production
system. Every design decision is classified in `docs/halp_method_spec.md`
as exact-from-paper, exact-from-official-blog, a required adaptation, a
material evaluation adaptation, or an unresolved ambiguity. See
`docs/halp_provenance.md` §3 for a corrected claim: an earlier draft of
this baseline used a linear (logistic-regression) reward model and
described it as faithful; this was inaccurate against the primary source
(the production reward model is a two-layer MLP) and has been fixed.

### Diagnostics exposed

- `n_cold_start_evictions`, `n_model_ranked_evictions`, `model_trained`
  (`HALPPolicy.diagnostics_summary()`).
- Per-step `CacheEvent.diagnostics["mode"]` ∈
  `{hit, direct_admit, cold_start_lru, model_ranked}`.

### Running

```bash
python -m lafc.runner.run_policy \
  --policy halp \
  --trace data/example_unweighted.json \
  --capacity 3 \
  --halp-training-trigger 10000 --halp-hidden-units 8 --halp-alpha 1e-4 --halp-seed 0
```

Full external-baseline comparison across all 7 manuscript trace families
and capacities 32/64/128:

```bash
python scripts/experiments/run_halp_comparison.py
```

Outputs write to `analysis/external_learned_baselines/halp/` — canonical
`*_heavy_r1`, `lrb/`, and `three_l_cache/` artifacts are never touched.

---

## Baseline 9: CACHEUS (Rodriguez et al., FAST 2021)

### Paper citation

Rodriguez, L. V., Yusuf, F., Lyons, S., Paz, E., Rangaswami, R., Liu, J.,
Zhao, M., & Narasimhan, G. (2021). **Learning Cache Replacement with
CACHEUS**. 19th USENIX Conference on File and Storage Technologies
(FAST '21).

### Official code source and pinned commit

[`sylab/cacheus`](https://github.com/sylab/cacheus) (author-released; no
LICENSE file — see `docs/cacheus_provenance.md`), commit
`1eec63ce166502be33ddd1f35bc041ed73a24f4d` (fetched 2026-08-06). Because
there is no license, the official source is fetched **externally** by
`scripts/setup/fetch_cacheus_official.py` into `external/` (gitignored,
never committed) and executed unmodified from there — it is not
reimplemented anywhere in this repository. See
`docs/cacheus_method_spec.md` for the full specification.

### Implemented policy names

- `cacheus` (official-source wrapper; requires running
  `python scripts/setup/fetch_cacheus_official.py` first — raises a clear,
  explicit error otherwise, no silent fallback to any other policy).

### Exact implemented variant

The authors' own `Cacheus` class (`code/algs/cacheus.py`), which **is**
the SR-LRU + CR-LFU expert combination — the paper's own headline
configuration, predeclared here before any evaluation (see
`docs/cacheus_method_spec.md`, "Primary variant used here"). All
hyperparameters (`initial_weight=0.5`, `history_size=cache_size // 2`,
initial `learning_rate=sqrt(2*ln(2)/cache_size)`, RNG seed 123) are the
official, unmodified defaults.

### Unit-size specialization: none needed

Unlike LRB, 3L-Cache, and HALP (all originally byte-capacity/CDN-oriented
and requiring a disclosed unit-size evaluation adaptation), **official
CACHEUS has no size/weight concept at all** — it is natively a unit-size,
object-slot paging algorithm (per its own README: "designed for paging
domain"). This repository's capacity semantics (32/64/128 object slots)
match the official algorithm's native domain exactly, with no adaptation.

### Faithfulness assessment

This is the authors' own, unmodified source code, not a reimplementation
— the strongest fidelity position of this repository's four external
learned baselines. The only disclosed deviations are non-algorithmic: a
portability-only patch (two empty `__init__.py` files added to the
*external, non-vendored* clone so it can be imported as a library) and an
explicit `ValueError` guard for capacity < 2 (an upstream crash in the
official source at that boundary, confirmed empirically and documented in
`docs/cacheus_method_spec.md`, "Known upstream limitation: capacity 1" —
not patched, and irrelevant to this repository's 32/64/128 capacities).
The official RNG seed (123) is hardcoded in the upstream source and is not
repository-controlled, unlike every other baseline here.

### Diagnostics exposed

- `final_weight_srlru`, `final_weight_crlfu`, `final_learning_rate`,
  `n_history_hits_lru`, `n_history_hits_lfu`, `dem_count`, `nor_count`
  (`CacheusPolicy.diagnostics_summary()`).
- Per-step `CacheEvent.diagnostics["mode"]` ∈ `{hit, official_cacheus}`.

### Running

```bash
# One-time fetch of the official source (external, gitignored, not vendored):
python scripts/setup/fetch_cacheus_official.py

python -m lafc.runner.run_policy \
  --policy cacheus \
  --trace data/example_unweighted.json \
  --capacity 3
```

Full external-baseline comparison across all 7 manuscript trace families
and capacities 32/64/128:

```bash
python scripts/experiments/run_cacheus_comparison.py
```

Outputs write to `analysis/external_learned_baselines/cacheus/` —
canonical `*_heavy_r1`, `lrb/`, `three_l_cache/`, and `halp/` artifacts are
never touched.

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
- `lrb` (Baseline 6 target — Song et al. 2020, NSDI, external learned baseline;
  requires optional `lightgbm` dependency)
- `three_l_cache` (Baseline 7 target — Zhou et al. 2025, FAST, external learned baseline;
  requires optional `lightgbm` dependency)
- `halp` (Baseline 8 target — Song et al. 2023, NSDI, external learned baseline;
  no official code exists, see `docs/halp_provenance.md`)
- `cacheus` (Baseline 9 target — Rodriguez et al. 2021, FAST, external learned
  baseline; runs the authors' own official source, see `docs/cacheus_provenance.md`
  — requires `python scripts/setup/fetch_cacheus_official.py` first)

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
