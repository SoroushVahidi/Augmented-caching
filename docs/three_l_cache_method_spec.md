# 3L-Cache method specification

## Sources

1. **Paper**: Wenbin Zhou, Zhixiong Niu, Yongqiang Xiong, Juan Fang, Qian Wang.
   "3L-Cache: Low Overhead and Precise Learning-based Eviction Policy for
   Caches." 23rd USENIX Conference on File and Storage Technologies (FAST
   25), pp. 237–254, Feb 2025. https://www.usenix.org/conference/fast25/presentation/zhou-wenbin
2. **Official artifact**: [`optiq-lab/3L-Cache`](https://github.com/optiq-lab/3L-Cache)
   (GPL-3.0), pinned commit `134cd159b635cdab75419a4281bed1a330fef31f`
   (fetched 2026-08-06) — confirmed as the authoritative, most-up-to-date
   repository by the paper's own Artifact Appendix ("The artifact is
   publicly available on GitHub at ... optiq-lab/3L-Cache. The latest
   version of the source code is hosted on the main branch."). A second
   repo, `admin333-paper/3L-Cache`, is the pre-acceptance double-blind
   submission mirror with near-identical content; not used as the reference.
   Core files consulted: `3LCache/TLCache.h`, `3LCache/TLCache.cpp`,
   `3LCache/TLCache_Interface.cpp`.
3. **Relationship to libCacheSim**: 3L-Cache is implemented as a cache
   policy (`TLCache`) inside a fork of [`1a1a11a/libCacheSim`](https://github.com/1a1a11a/libCacheSim)
   (the artifact repo *is* a libCacheSim fork with `3LCache/` added), and
   has since been proposed for upstream inclusion
   (`1a1a11a/libCacheSim#119`, "Adding 3L-Cache"). Not derived from LRB's
   codebase, though it shares LRB as a direct architectural predecessor
   (object-level learning, GBM, log1p(reuse-interval) target) and the paper
   explicitly benchmarks against LRB.
4. **License**: GPL-3.0. No source was copied verbatim into this repository
   (independent reimplementation from the documented algorithm); noted for
   citation/compatibility purposes.

## What "3L" means

Not "three-layer" or "three-level" — the abstract spells it out: an
object-level learning policy with **L**ow computation overhead, the
**L**owest object miss ratio, and the **L**owest byte miss ratio among
learning-based policies. Three "L"s.

## Method specification, classified

| Design point | Spec | Classification |
|---|---|---|
| Feature vector (6 total) | age (time since last request), up to 3 most-recent inter-arrival deltas (most-recent-first, NaN-padded), size, frequency (`_freq`, a monotonically-incrementing per-object request counter reset only when the object's metadata is destroyed/recreated, capped at 65535) | exact from code (`TLCache.h` `n_feature = max_n_past_timestamps(4) + 2`) |
| Prediction target | `log1p(time-to-next-request)`, GBM regression, L2 | exact from paper §4.2.2 + code |
| Model / hyperparameters | LightGBM GBDT: `num_iterations=16, num_leaves=32, learning_rate=0.1, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5` | exact from code (`TLCache.h:295-306`) — note `num_iterations=16`, half of LRB's 32; not stated explicitly in paper prose |
| Training batch size | `M = 65536` (`131072/2`) | exact from code; matches paper's stated `M=64K` default |
| Sampled-label subsampling | A pending sample only actually becomes a training row with **25% probability** (`rng % 4 == 0`), or always if no model has been trained yet | exact from code (`TLCache.cpp:93`, justified in a code comment: "a probability greater than 25% has little impact on cache efficiency" — this specific number is not in the paper's main text) |
| Delayed-label construction (re-request) | `log1p(current_seq − sample_time)`, i.e. the observed gap | exact from code, same mechanism as LRB |
| Delayed-label construction (window-exit) | `log1p(B + (current_seq − meta.past_timestamp))`, where `B` is a **running empirical maximum** ("MAX_EVICTION_BOUNDARY"), *not* a fixed constant like LRB's `2×window`: `B` is frozen at each retrain from the largest wait-until-exit observed since the *previous* retrain | exact from code (`TLCache.cpp:48-49, 140-142`) — matches paper prose ("waiting time plus the longest waiting time recorded among the training samples") but the paper prose alone does not specify the freeze/snapshot timing; resolved from code |
| Score inverse-transform | `exp(score)` (not `expm1(score)`) to invert the `log1p` training target | exact from code (`TLCache.cpp:451,461`) — a real asymmetry in the official code (log1p forward / plain exp backward, off by a constant +1 in linear space); reproduced faithfully, not "corrected" |
| Score/heap comparability offset | `byte_miss_ratio` objective adds `(current_seq − origin_current_seq)` — elapsed time since the model was last (re)trained — to every predicted absolute eviction time, so that predictions gathered across multiple sampling rounds under the same trained model remain comparable on one shared heap; `object_miss_ratio` objective does **not** apply this offset (score = `size × exp(score)` only) | exact from code (`TLCache.cpp:448-469`) — a genuine asymmetry between the two objective modes, not size-invariant like LRB's analogous mechanism; **required a real choice for this repo** (see below) |
| Sliding window ("out-cache" retention) | Dynamic, tied to *current* in-cache occupancy: `max_out_cache_size = |in_cache| × (hsw − 1) + 2`, not a fixed request-count window like LRB's `memory_window` | exact from code (`TLCache.cpp:131`) — a structural difference from LRB worth naming explicitly |
| `hsw` auto-tuning | Starts at 2, capped at 6; grows by 1 after a retrain if window-hit-rate minus in-cache-hit-rate, normalized, exceeds 1%, evaluated only once `n_req > 10^6` and the window has filled at least once | exact from code (`TLCache.cpp:52-59`) |
| Eviction-candidate generation ("bidirectional sampling") | Two independent sampling passes per resampling round: (a) **from the head** — `quick_demotion()` draws from a FIFO of recently-admitted object keys as long as accumulated new-object bytes exceed `currentSize × Q/100`; (b) **from the tail** — a *persistent* pointer walks the in-cache LRU list (continuing across calls, not restarting each time), sampling objects with `freq < f` OR still within the leading `x%` of one full lap, up to `n = min(1024, 1%·L + 2)` per call | exact from code (`TLCache.cpp: rank(), quick_demotion()`) |
| Auto-tuning of `f`, `x`, `Q` | Recomputed once per **full lap** of the tail-scan pointer around the queue, from accumulated eviction-outcome histograms (`f` via 99th-percentile interpolation over a `log2(frequency)` histogram of evicted objects; `x`/`Q` via simple increment/decrement comparisons of tail-scan vs. new-object eviction counts) | exact from code (`TLCache.cpp:233-261`) — matches Table 2's rules; exact formulas resolved from code since the paper's Table 2 alone is not fully unambiguous (e.g. the percentile-interpolation formula for `f`) |
| Batched eviction | Eviction candidates are scored once per resampling round and pushed onto a shared min-heap (ordered by predicted absolute eviction time); `evict_nums = |sampled| / eviction_rate` (`eviction_rate` is a **fixed internal constant = 2**, despite superficially resembling an auto-tuned quantity) evictions are then drawn from the heap before the next resampling round | exact from code |
| Heap staleness ("stale-entry validation") | A `pred_map[key] → predicted_value` side table is checked on every heap pop; if the object was re-requested or already evicted since being scored (`pred_map` entry missing or value changed), the stale heap entry is discarded and the next one is popped instead | exact from code — essential to the batched design's correctness, not optional |
| Retrain gating | Retrain triggers at `labeled_rows ≥ batch_size`, **but only if `evict_nums ≤ 0`** (not mid-way through drawing down an active eviction batch) | exact from code (`TLCache.cpp:96,145`) — a real difference from LRB, which retrains unconditionally at the threshold |
| Cold start | Before any model is trained: plain LRU eviction (queue head), no sampling/prediction | exact from paper (footnote 3, §4.2.1: "Since model training requires a substantial amount of data, we initially employ LRU until the model is trained") + code |
| Tail-scan pointer initial position | First-ever call starts scanning from the current LRU head (oldest resident object) | **repository-required adaptation**: the official code defaults `samplepointer = 0`, a raw array index into `in_cache.metas`, which by the time of first use (after cold-start evictions have already swap-removed and relocated objects) points at whichever object an unrelated eviction happened to move into slot 0 -- an artifact of array-based storage with no equivalent meaning in this key-based port. Starting from the LRU head is the well-defined, clearly-intended interpretation of "sampling from the tail," found and fixed during implementation (an earlier draft left the pointer uninitialized, silently producing zero tail-scan candidates on the first resampling round -- covered by `test_tail_scan_pointer_initializes_on_first_use`) |
| Object sizes | Native design is byte-size-aware (`_size` feature, `objective` byte/object modes, `reserved_space`/`sample_boundary` interacting with byte accounting) | **repository-required adaptation** — see Strategy B below |
| Randomness / seeds | One `std::default_random_engine`, used for both the 25%-subsampling coin flip and the per-request unlabeled-sample draw | mechanism exact from code; this repo pins an explicit seed (`random.Random(seed)`, LightGBM seeds pinned), matching the deterministic-seed adaptation already established for LRB |
| Known reference-code defect | `quick_demotion()` declares `int i, j = 0;` — `i` is **never initialized** before use as a loop counter (`i < new_obj_keys.size()`), which is undefined behavior in the reference C++ | **unresolved ambiguity in the official source**, resolved here via the clearly-intended semantic (`i` starts at 0, consistent with the subsequent `new_obj_keys.erase(begin, begin+i)` call implying a front-counted scan) — documented, not silently guessed |
| Metric | Both byte-miss-ratio and object-miss-ratio reported in the paper; under this repo's unit-size specialization the two coincide numerically for the size-multiplication term, but **not** for the heap-comparability offset (see above) | exact match for the metric itself; the objective-mode choice is a real decision (see below) |

## Compatibility with this repository's manuscript evaluation (Strategy B)

Same conclusion as for LRB (see `docs/lrb_method_spec.md`): the KBS manuscript
evaluates standard **unweighted paging** — unit miss cost, capacity in
object slots (32/64/128), primary metric = request misses/miss-ratio, no
reliable per-object byte sizes wired through this repository's simulator.
3L-Cache's native design assumes variable-size, byte-capacity objects
throughout (feature `size`, `objective` byte/object modes, `Q`/`reserved_space`
interacting with byte thresholds). → **Strategy B: 3L-Cache adapted to the
unit-size paging setting**, not a reproduction of the paper's byte-cache
CDN/block-storage evaluation. Every object's size is held at a constant 1.

**Objective-mode decision, made explicitly and disclosed (not silently
picked to flatter either method's numbers)**: under size≡1, the
`sizes[i] × exp(score)` term is identical either way, but the two modes'
*heap-comparability offset* is not — `byte_miss_ratio` adds
`(current_seq − origin_current_seq)`, `object_miss_ratio` does not. This
repository uses `objective="byte_miss_ratio"` (the official code's own
default) because that branch is the mechanistically complete one (it keeps
predictions gathered at different times on the same heap comparable);
`object_miss_ratio` in the reference omits this normalization, which reads
as an incompleteness in the original code rather than an intentional
design choice for the object-count-optimized path. This is recorded here so
the choice is auditable, not hidden inside the implementation.

## Fairness with the LRB and evict_value_v1 protocol

Identical experimental conditions are required by design: same trace files
(`analysis/wulver_trace_manifest_full.csv`), same preprocessing
(`load_trace_from_any`), same capacities (32/64/128), same 50,000-request
budget, same no-warmup-exclusion convention, same metric (total misses /
hit rate). See the fairness table in the final report and
`scripts/experiments/run_three_l_cache_comparison.py`.

## Known limitations

- Auto-tuning (`h_sw, f, x, Q, n`) is implemented from the exact code-level
  formulas, but this is meaningfully more stateful machinery than LRB's;
  algorithmic-level parity here is validated via targeted unit tests on
  small scripted traces (hand-verified trigger conditions and updated
  values), not a full-scale side-by-side replay against a built C++ binary
  (same rationale as LRB: building `optiq-lab/3L-Cache` requires a full
  libCacheSim + LightGBM C API + CMake toolchain, judged impractical for
  this sandbox).
- The reference source's uninitialized-`i` defect in `quick_demotion()` has
  no single unambiguous "correct" fix; the interpretation used here is
  reasonable and documented but is, by definition, not something that can
  be verified against the reference's actual (undefined) runtime behavior.
- Model-level (not algorithmic-level) exact score agreement with the
  reference cannot be guaranteed across LightGBM versions/builds, as with
  LRB.
