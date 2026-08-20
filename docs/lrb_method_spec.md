# LRB (Learning Relaxed Belady) — method specification

## Sources

1. **Paper**: Zhenyu Song, Daniel S. Berger, Kai Li, Wyatt Lloyd. "Learning
   Relaxed Belady for Content Distribution Network Caching." NSDI 2020,
   pp. 529–544.
2. **Official implementation**: [`sunnyszy/lrb`](https://github.com/sunnyszy/lrb)
   (BSD-2-Clause), pinned at commit `9e8b4423383c01c4528deb447f152f0437a37c3a`
   (fetched 2026-08-06). Primary files consulted:
   `include/webcachesim/caches/lrb.h`, `src/caches/lrb.cpp`.

Per the task's instructions, the official code is treated as authoritative
wherever it disambiguates something the paper describes only informally.

## License compatibility

`sunnyszy/lrb` is BSD-2-Clause. No source code was copied verbatim into this
repository — the Python implementation in `src/lafc/policies/lrb.py`,
`src/lafc/lrb_features.py`, and `src/lafc/lrb_model.py` is an independent
reimplementation of the documented algorithm, grounded in but not derived
line-by-line from the C++ source. BSD-2-Clause is compatible with reuse under
this repository's own license terms.

## Implementation status

**Native, unit-size specialization (Strategy A).** Not a byte-cache
reproduction, not an adapter to the official C++ simulator. See
`src/lafc/policies/lrb.py` (registered as CLI policy `lrb`), and
`docs/baselines.md` Baseline 6 for the standard baseline write-up format.

## Method specification, classified

Every row states the design decision, its concrete form here, and whether it
is exact-from-paper, exact-from-code, a required adaptation, or an optional
deviation.

### Online cache state

- **In-cache metadata** (`LRBPolicy._in_cache_meta`, `_lru_queue`): one
  `ObjectMeta` per physically-cached object, an LRU queue for the cold-start
  and window-boundedness fallback. *Exact from code* (`InCacheMeta`,
  `in_cache_lru_queue` in `lrb.h`).
- **Ghost metadata** (`_ghost_meta`, `_ghost_expiry`): metadata for objects
  evicted from the physical cache within the last `memory_window` requests,
  kept only so their eventual re-request or window-timeout can still produce
  a training label. *Exact from code* (`out_cache_metas`,
  `negative_candidate_queue`).
- Data structure: Python dict + `collections.OrderedDict` in place of the
  official code's index-swapped `vector`s and a circular-buffer ring for
  `past_distances`. *Adaptation* — same semantics, simpler structures; no
  behavioral difference (verified: `past_distances` stays most-recent-first
  and capped identically either way).

### Admission and eviction ordering

- The official C++ implementation inserts the newly-fetched object into
  `in_cache_metas` **before** evicting (`admit()` calls `evict()` in a loop
  only if now over capacity), meaning the just-fetched object is technically
  eligible to be sampled — and in principle immediately evicted — as one of
  its own eviction candidates.
- This repository's `BasePolicy`/`CacheState` contract requires the opposite
  order (evict before add; `CacheState.add` raises if the cache is already
  full). **Required adaptation**: eviction happens strictly before admission
  here, so the just-fetched page can never be its own eviction candidate.
  This matches this repository's existing convention (see
  `offline_belady.py`'s explicit `exclude` parameter) but is a real,
  documented behavioral difference from the literal reference. Under
  unit-size capacities this can only matter in the (already rare in the
  original design) case where a freshly-inserted object would otherwise have
  been sampled and evicted in the same step.

### Candidate features (44 total under default config)

Feature-row layout, *exact from code* (`lrb.h:32-43`, `MetaExtra`,
`TrainingData::emplace_back`, `LRBCache::rank`):

| Index | Feature | Notes |
|---|---|---|
| 0 | age = sample_timestamp − past_timestamp | |
| 1–31 | past request deltas, most-recent-first | unfilled slots = NaN ("missing"), matching the official sparse-CSR omission |
| 32 | object size | **constant 1.0 under unit-size specialization** |
| 33 | `n_within` — count of leading deltas whose running sum stays < `memory_window` | |
| 34–43 | 10 EDCs (exponentially-decayed request-rate counters at windows 2¹⁰…2¹⁹) | `EDC_i ← EDC_i · 2^(−Δ₁/2^(9+i)) + 1`; first-ever update seeds `EDC_i = hash_edc[idx] + 1` (i.e. as if the prior value were 1, not 0) — exact detail of `MetaExtra`'s constructor, not the paper's simplified prose formula |

`n_extra_fields` (up to 4 CDN categorical fields in the official code, e.g.
object type) is fixed at **0** here: *required adaptation* — this
repository's trace format carries no such fields.

### Object sizes — unit-size specialization

Every `ObjectMeta.size = 1.0`. The size feature and the
`objective="object_miss_ratio"` score-multiplication step (`rank()`
multiplies scores by size only when `objective == object_miss_ratio`) are
both kept structurally rather than removed, but are numerically inert
(multiplying by 1). *Required adaptation* per Strategy A — see "Fairness
protocol" in `docs/baselines.md`.

### Eviction-candidate sampling

On a full-cache miss:
1. Check the LRU-tail object. If no model has been trained yet, **or** its
   age ≥ `memory_window`, evict it directly via plain LRU — no model call.
   *Exact from paper §4.1* ("our implementation uses LRU as a fallback until
   sufficient training data is available") *and code* (`rank()`'s first
   block). This is the one documented, paper-sanctioned fallback used here;
   there is no other silent fallback anywhere in this policy.
2. Otherwise, sample `sample_rate` (default 64) **distinct** in-cache
   objects uniformly at random (`random.Random.sample`, in place of the
   official code's rejection-sampling-until-distinct loop — *adaptation*,
   provably the same distribution), score all with one batched model call,
   and evict the candidate with the **largest** predicted
   `log1p(time-to-next-request)`. *Exact from paper §4.4/§4.5 + code.*
3. Deterministic tie-break by smallest `page_id`. *Adaptation* — the
   official code's `std::sort` with `>` is not guaranteed stable, so exact
   reference tie-break behavior is undefined; this repository picks one
   fixed, documented rule for reproducibility.

### Training-label construction (delayed labels, no future leakage)

Predict `log1p(time-to-next-request)`. A sampled candidate's label matures
only from already-observed events:

| Event | Label | Source |
|---|---|---|
| Re-requested at time `t` (sampled at `s`) | `log1p(t − s)` | *exact from code*, `lookup()` |
| Force-evicted while still resident, age ≥ `memory_window` | `log1p(age + memory_window)` | *exact from code*, `evict()` |
| Ghost entry times out of the window (never re-requested) | `log1p(2 × memory_window)` | *exact from code*, `forget()`; also *exact from paper §4.2* ("LRB assigns a label as 2× the window size") |

The feature row used at maturation reflects the object's state **as of the
original sample time**, not the current time — maturation always happens
before that object's own metadata is updated for the current request,
mirroring the official code's explicit ordering comment ("make this update
after update training, otherwise the last timestamp will change"). This is
the structural mechanism that guarantees no ground-truth future information
(`Request.actual_next`) is ever read; see `test_no_future_leakage_from_actual_next_or_predicted_next`
in `tests/test_lrb.py`, which corrupts `actual_next`/`predicted_next` for
every request and confirms the policy's decisions are unchanged.

### Sliding memory window

Circular-buffer window (`t % memory_window`) bounding how long ghost
metadata is retained. *Mechanism exact from code.* The paper tunes this
per-trace on a 20% validation prefix (§4.1, §6.6); the official code's
constant default (67,108,864 requests) is CDN-scale and never fires within
this repository's ≤50,000-request/trace, 32–128-slot evaluation. **Required
adaptation**: this repository ships an illustrative default
(`memory_window=4096`) and performs its own small validation-only grid
search per trace/capacity in `scripts/experiments/run_lrb_external_baseline.py`,
analogous in spirit to the paper's own protocol — never tuned against the
evaluated/test region.

### Model family and hyperparameters

GBDT regression (L2 loss), LightGBM. *Exact from code*
(`lrb.h:491-503`): `num_iterations=32, num_leaves=32, learning_rate=0.1,
feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5`.

**Optional deviation**: the official code relies on LightGBM's stock RNG
defaults for bagging/feature-fraction sampling during training, without
pinning them. This repository explicitly pins `seed`/`bagging_seed`/
`feature_fraction_seed` and sets `deterministic=True`,
`force_row_wise=True` (`src/lafc/lrb_model.py`), to satisfy this
repository's deterministic-seed requirement. Strictly safer, not a change to
model family, objective, or any tuned hyperparameter.

### Training schedule

Full refit (not incremental) whenever the matured-label buffer reaches
`batch_size` (128K/131,072 in the official code). *Mechanism exact from
code.* **Required adaptation**: at this repository's 50,000-request/trace
scale, the official default never fires even once — the model would never
train, and the policy would silently degenerate to permanent cold-start LRU.
`batch_size` is therefore a validation-tunable parameter here too (default
`2048`, illustrative), with one explicit **untuned paper-default** run
(`memory_window=67108864, batch_size=131072`) included in the experiment
protocol specifically to make this degeneration visible rather than hidden.

### Score direction

Evict the candidate with the **largest** predicted `log1p(time-to-next-request)`
(farthest predicted reuse). *Exact from paper/code.*

### Warm-up / cold start

Before any model has been trained: plain LRU eviction, no model call, no
candidate sampling. *Exact from paper §4.1 and code* — the paper explicitly
describes this as the intended design, not an ad hoc addition.

### Randomness and seeds

One policy-owned `random.Random(seed)`, re-created in `reset()` (matches
this repository's `adaptive_query.py` convention), used for both the
per-request unlabeled-sample draw and the per-eviction candidate draw, in a
fixed order. LightGBM's internal seeds are pinned as noted above.
*Adaptation for reproducibility*, required by this repository's
deterministic-seed constraint.

### Metadata-memory accounting

The paper reports C++ byte-level overhead (Table 3: feature/sample
overhead in bytes per object). Not portable to a Python object model.
**Adaptation**: this repository exposes counts instead
(`n_in_cache_meta`, `n_ghost_meta`, `n_pending_rows`, `n_retrain`) via
`diagnostics_summary()`, not byte-level accounting. Documented as a
deviation, not silently dropped.

### Metrics

Request misses / miss ratio. Byte-miss ratio is not separately reported:
under unit-size specialization it is numerically identical to request-miss
ratio (every object has size 1), so there is no separate quantity to
compute. *Exact match with this repository's manuscript protocol
(unweighted paging, unit miss cost) — no adaptation needed.*

### Parameters tuned on validation data vs. fixed defaults

- Paper: `memory_window` is the only hyperparameter tuned per trace (on a
  20% validation prefix); `sample_rate=64`, GBM hyperparameters, and
  `batch_size=128K` are fixed defaults, explicitly justified in the paper as
  past the point of diminishing returns (§4.3).
- This repository: `sample_rate=64` and the GBM hyperparameters are kept as
  fixed paper/code defaults (not tuned). `memory_window` and `batch_size`
  are validation-tuned per trace/capacity via a small grid search on a
  held-out validation prefix (never the evaluated/test region), because
  their official numeric defaults are meaningless at this repository's
  request-count scale (see above). This mirrors the paper's own
  validation-only tuning philosophy, extended to the one additional
  parameter (`batch_size`) that also needs rescaling for a much shorter
  trace.

### Simulator vs. Apache Traffic Server (ATS) prototype

The paper describes both a pure C++ simulator (used for all of the paper's
byte-miss-ratio comparison results) and a production ATS prototype sharing
the same core library but adding asynchronous I/O, a lock-free eviction
queue, and flash-layer emulation. **This port targets the algorithmic
simulator core only** — the ATS system-level machinery (I/O scheduling,
SSD write-amplification handling, etc.) is out of scope for a pure-Python
replay simulator and irrelevant to the caching-policy semantics being
evaluated here.

## Known limitations

- `memory_window`/`batch_size` defaults shipped in `LRBConfig` are
  illustrative, not validation-tuned; always use the experiment script's
  validation-tuning step (or explicit CLI overrides) for any reported
  comparison.
- Parity validation against the official implementation (see
  `tests/test_lrb.py`'s `test_edc_recurrence_matches_official_metaextra`,
  `test_n_within_matches_official_loop_semantics`, and the label-formula
  tests) is algorithmic/feature/label-level, hand-derived against the pinned
  source above — not a literal binary-level comparison against a built
  `sunnyszy/lrb` C++ simulator, which requires CMake, the LightGBM C API,
  and the MongoDB C++ driver and was judged out of scope for this
  repository's Python-only sandbox.
- Model-level (not algorithmic-level) agreement with the official
  implementation cannot be established exactly even in principle, since
  exact floating-point GBDT training trajectories depend on the LightGBM
  build/version and are not guaranteed bit-identical across environments.
