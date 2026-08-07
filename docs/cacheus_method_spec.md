# CACHEUS — method specification

## Sources

1. **Paper**: Liana V. Rodriguez, Farzana Yusuf, Steven Lyons, Eysler Paz,
   Raju Rangaswami, Jason Liu, Ming Zhao, Giri Narasimhan. **"Learning
   Cache Replacement with CACHEUS."** 19th USENIX Conference on File and
   Storage Technologies (FAST '21), 2021.
   https://www.usenix.org/conference/fast21/presentation/rodriguez
2. **Project page**: FIU Systems Research Laboratory,
   https://sylab-srv.cs.fiu.edu/doku.php?id=projects:cacheus — links to
   the public software repository below.
3. **Official source** (author-released, primary/authoritative for every
   implementation-level detail): https://github.com/sylab/cacheus,
   owner `sylab` (matches the FIU lab's GitHub account and the repository
   description, which links directly to the authors' own USENIX
   presentation page), default branch `main`, pinned commit
   `1eec63ce166502be33ddd1f35bc041ed73a24f4d` (fetched 2026-08-06 via
   `scripts/setup/fetch_cacheus_official.py`). **No LICENSE file** (GitHub
   license API returns 404; `license` field is `null`) — see
   `docs/cacheus_provenance.md`.
4. Files consulted directly: `code/algs/cacheus.py` (the `Cacheus` class —
   the sole implementation file for this baseline), `code/algs/lru.py`
   (used only for cross-simulator parity validation, not the baseline
   itself), `code/algs/lib/{dequedict,heapdict,cacheop,optional_args,
   pollutionator,visualizinator}.py`, `code/run.py`, `code/run_alg.py`,
   `code/get_algorithm.py`, `code/example.config`, `README.md`.

Authority level: **author-released official source**, the highest
available. Every detail below is classified against it directly (not
against a secondary summary).

## Integration strategy: official-source wrapper

`src/lafc/policies/cacheus.py` (`CacheusPolicy`) drives the authors'
`Cacheus` class from the pinned commit **unmodified**, via
`src/lafc/cacheus_official_loader.py`. The official source is fetched
externally by `scripts/setup/fetch_cacheus_official.py` into
`external/cacheus_official/` (gitignored, never committed — see
`docs/cacheus_provenance.md` for why). No SR-LRU/CR-LFU algorithm code is
reimplemented anywhere in this repository; the adapter only translates
between this repository's `BasePolicy`/`Request`/`CacheEvent` interface and
the official class's `(cache_size, window_size, **kwargs)` /
`.request(oblock, ts) -> (CacheOp, evicted_oblock)` interface, and mirrors
its hit/miss/eviction decisions into this repository's own `CacheState` for
bookkeeping consistency with every other policy here.

**Note on file layout**: this baseline does *not* have separate
`sr_lru.py` / `cr_lfu.py` modules. In the authors' own code, "SR-LRU" and
"CR-LFU" are not two decoupled, independently instantiable policy objects
— they are two competing eviction-candidate-proposal mechanisms (an
LRU-ordered `Q` stack and an LFU min-heap) integrated directly inside one
`Cacheus` class, combined via LeCaR-style regret-weighted random selection.
A file structure with standalone `sr_lru.py`/`cr_lfu.py` modules would
therefore misrepresent the official implementation's actual structure.

## Method specification, classified

Fidelity classification: **(1) exact from paper**, **(2) exact from
official source**, **(3) official source differs from paper**,
**(4) integration-only adaptation**, **(5) material evaluation
adaptation**, **(6) unresolved ambiguity**.

### CACHEUS expert-combination framework

CACHEUS is a LeCaR-style adaptive combiner over two experts: an LRU-like
proposal and an LFU-like proposal, selected by weighted random sampling and
re-weighted via multiplicative-weights (Hedge) updates driven by delayed
feedback. *(1)/(2) exact.*

### Primary variant used here: SR-LRU + CR-LFU

The `Cacheus` class **is** the SR-LRU + CR-LFU combination — there is no
separate class to instantiate for "the primary variant" versus "an
alternative expert pair." The paper's alternative pairings (ARC+LFU,
LIRS+LFU, via `alecar6.py`, `arcalecar.py`, `lirsalecar.py` in the official
repo) are **not used** in this baseline: SR-LRU+CR-LFU is predeclared here,
before any evaluation, as the paper's own headline configuration ("CACHEUS
using the newly proposed lightweight experts, SR-LRU and CR-LFU, is the
most consistently performing caching algorithm across a range of workloads
and cache sizes" — official repo README / project description). *(1)/(2)
exact; predeclared per this task's requirement not to select an expert
pair after seeing results.*

### SR-LRU semantics (as integrated in `Cacheus`)

- Two deques, `S` (`self.s`) and `Q` (`self.q`), partition the physical
  cache. New/recently-demoted items live in `Q`; items that have proven
  themselves (via a `Q`-hit or history-hit promotion) move to `S`.
  *(2) exact, `code/algs/cacheus.py` `hitinQ`, `addToS`/`addToQ`.*
- `S`/`Q` sizes are **adaptively repartitioned**, not fixed: `adjustSize()`
  grows `s_limit` when hits occur in `Q` (proportional to
  `nor_count / dem_count`) and grows `q_limit` on ghost-history misses
  (proportional to `dem_count / nor_count`) — an ARC-style adaptive
  partition, which is the mechanism that makes this variant "scan
  resistant": a scan (a burst of one-time, never-repeated accesses) mostly
  churns through `Q` without displacing the proven working set in `S`.
  *(2) exact, `adjustSize`.*
- Initial partition: `q_limit = max(1, round(0.01 * cache_size))`,
  `s_limit = cache_size - q_limit`. *(2) exact, `hirsRatio = 0.01`.*
- LRU-proposal candidate for eviction: the oldest entry in `Q`
  (`getLRU(self.q)`). *(2) exact.*

### CR-LFU semantics (as integrated in `Cacheus`)

- A single shared min-heap over **all** physically-cached entries
  (`self.lfu = HeapDict()`), keyed by `(freq, -time)` via
  `Cacheus_Entry.__lt__` (lower frequency first; among equal frequency,
  the **more recently** touched entry sorts as "greater," i.e. the
  least-recently-used among equal-frequency entries is preferred as the
  LFU-proposal victim). *(2) exact.*
- LFU-proposal candidate for eviction: the heap minimum
  (`getHeapMin()`). Frequency increments on every hit (`x.freq += 1`,
  `hitinS`/`hitinQ`). *(2) exact.*
- "Churn resistance" comes from combining this with the ghost-history
  delayed-feedback mechanism (below): a page that gets evicted by the LFU
  proposal but is needed again soon generates a penalty against the LFU
  expert's weight, damping how often churny (frequency-inflated-then-cold)
  pages get to dominate eviction decisions. *(3) — this causal explanation
  is this repository's inference from the code's mechanics, not a direct
  quote from the paper; the paper's own "churn-resistant" framing is
  paraphrased from the README/abstract, not independently re-derived here.*

### Ghost/history structures and delayed feedback

- Two ghost histories, `lru_hist` and `lfu_hist` (`DequeDict`), each capped
  at `history_size = cache_size // 2` (official default; overridable via
  `CacheusConfig.history_size`). *(2) exact.*
- On eviction, the evicted entry is recorded into the history
  corresponding to **which expert's proposal was taken** (policy 0 → LRU
  history, policy 1 → LFU history; policy -1, the agreement case below,
  records into neither — `addToHistory(x, policy)` with `policy=-1`
  returns immediately). *(2) exact.*
- **Delayed feedback**: if a page later reappears and is found in
  `lru_hist`, this is treated as a *miss* (`CacheOp.INSERT`, since the page
  is not in the physical cache) but triggers `adjustWeights(-1, 0)` — a
  penalty against the LRU/SR-LRU expert's weight, because that expert's
  earlier eviction decision was, in hindsight, wrong. Symmetrically, an
  `lfu_hist` hit triggers `adjustWeights(0, -1)`, penalizing CR-LFU.
  *(2) exact, `hitinLRUHist`/`hitinLFUHist`.*
- **Feedback expiration**: if a page is evicted from a ghost history
  before it reappears (history is capped at `history_size`; the oldest
  entry is dropped when a new one arrives at full capacity), no feedback
  is ever generated for it — the responsible expert's weight is simply
  never adjusted for that decision. *(2) exact, `addToHistory`'s
  `if len(policy_history) == self.history_size: evicted = ...; del
  policy_history[evicted.oblock]`.*

### Expert victim proposal and selection (`evict()`)

- Both proposals are computed every eviction: `lru = getLRU(self.q)`,
  `lfu = getHeapMin()`. *(2) exact.*
- **Agreement case**: if `lru is lfu` (the same entry object — both
  experts propose the identical victim), it is evicted directly with
  `policy = -1`; no random draw occurs and no weight update follows (there
  was no actual choice made). *(2) exact.*
- **Disagreement case**: a weighted Bernoulli draw,
  `getChoice() = 0 if np.random.rand() < W[0] else 1`, selects which
  proposal is followed (`policy=0` → evict the `Q`-LRU candidate,
  `policy=1` → evict the LFU-heap-min candidate). *(2) exact.*
- Randomness is process-global `numpy.random`, seeded to a **fixed
  constant, 123**, inside `Cacheus.__init__` (`np.random.seed(123)`) —
  every fresh `Cacheus` instance resets numpy's *global* RNG state to seed
  123. *(2) exact — preserved as-is (an algorithm-changing patch to
  "fix" this would deviate from official behavior); disclosed side effect:*
  instantiating `Cacheus` resets any other code in the same Python process
  that relies on `numpy`'s global RNG state. Not a practical risk for this
  repository's runners (one policy instantiated and run to completion per
  row, no other numpy-global-RNG-dependent code interleaved with a live
  `Cacheus` instance), but documented per this task's leakage/determinism
  rigor requirements.

### Weight update equation

- Hedge/multiplicative-weights update:
  `W = W * exp(learning_rate * reward)`, then renormalized
  (`W = W / sum(W)`), then clamped so neither weight can reach exactly 0
  or 1 (`if W[0] >= 0.99: W = [0.99, 0.01]`, symmetric for `W[1]`).
  *(2) exact, `adjustWeights`.* `reward` is always one of `(-1, 0)` or
  `(0, -1)` (a penalty on the expert whose eviction proved wrong; the
  other expert's weight is untouched, i.e. multiplied by `exp(0) = 1`)
  — there is no positive reward term; this is a pure-penalty (regret)
  scheme, not a reward-for-correct-decisions scheme. *(2) exact.*

### Learning-rate schedule

- Initial learning rate: `sqrt(2 * ln(2) / cache_size)`
  (`Cacheus_Learning_Rate.__init__`, `period_length = cache_size`,
  **not** the separate `window_size` constructor argument — `window_size`
  is used only by the visualization subsystem, see below). *(2) exact.*
- Adapts every `cache_size` requests (`period_length`): compares the
  change in hit-rate to the change in learning rate over the last period
  and moves the learning rate in whichever direction most recently
  correlated with an *improving* hit rate; if hit-rate improvement has
  stalled for 10 consecutive periods, resets to
  `learning_rate_reset = clip(initial_learning_rate, 0.001, 1)`; otherwise
  perturbs randomly (again via global `numpy.random`) when stuck at a
  learning-rate extreme. *(2) exact, `Cacheus_Learning_Rate.update`,
  `updateInDeltaDirection`, `updateInRandomDirection`.*

### Cold-start behavior

There is no separate "cold start uses plain LRU" phase, unlike this
repository's LRB/3L-Cache/HALP baselines: `Cacheus` is adaptive from
request 1, starting from a **uniform prior** `W = [0.5, 0.5]`
(`initial_weight = 0.5`) and empty ghost histories, so early eviction
decisions are unweighted coin flips between the two proposals (or the
single agreed-upon proposal when they coincide) until enough delayed
feedback has accumulated to differentiate the experts. *(2) exact.*

### Admission

Every miss is admitted (to `S` if there is free `S` capacity and `Q` is
currently empty, else to `Q`, evicting first if the cache is already full)
— no admission bypass/filtering logic exists in `Cacheus.miss()` (unlike,
e.g., 3L-Cache's `_quick_demotion` filtering, which does not apply here).
*(2) exact.*

### Cache-size dependence and unit-size vs. variable-size support

`Cacheus_Entry` tracks only `(oblock, freq, time, is_new, evicted_time,
is_demoted)` — **no size/weight field at all**. `cache_size` is a plain
object-slot count throughout (`q_limit`, `s_limit`, `history_size`, the
learning-rate `period_length` all scale directly off it). The official
CACHEUS algorithm is **inherently a unit-size, paging-domain algorithm**
(the official README states this explicitly: "Cacheus is a novel cache
replacement algorithms designed for paging domain"). *(1)/(2) exact —
unlike LRB/3L-Cache/HALP (all originally byte-capacity/CDN-oriented and
requiring a disclosed unit-size evaluation adaptation), **no unit-size
adaptation is needed for CACHEUS at all**: this repository's capacity
(32/64/128 object slots) and unit-size semantics match the official
algorithm's native domain exactly.*

### Trace/object identity ("Semantic compatibility")

`oblock` is used only as a generic hashable dict/heap key throughout the
official source — never arithmetically manipulated, never assumed
numeric. This repository passes `Request.page_id` (a string) directly as
`oblock`, with no ID-mapping/conversion layer: **Strategy A (direct
official unit-size execution)**, not a trace-format adapter. `ts` (the
official `request(oblock, ts)` second argument) is used only by the
visualization subsystem (`Visualizinator.add`, guarded by
`enable_visual`, `False` by default) and has **zero effect on any
algorithmic decision** — verified by reading `Cacheus.request()`, where
the only use of the passed `ts` is inside `self.visual.add({...ts...})`
and `self.visual.addWindow({...}, self.time, ts)`, both no-ops when
`enable_visual=False`. This repository passes `request.t`. *(2) exact,
confirmed by direct code inspection, not inferred.*

### `window_size` (non-algorithmic)

Required positional constructor argument, used only to size the
`Visualizinator`'s windowed-statistics deques
(`Cacheus.__init__`→`self.visual = Visualizinator(..., window_size=
window_size, ...)`); irrelevant to any hit/miss/eviction outcome since
`enable_visual` defaults to `False`. The official `run.py` harness sets
`window_size = 100` whenever cache sizes are given as absolute integers
(not fractions of the working set) — exactly this repository's case (32,
64, 128) — so `CacheusConfig.window_size` defaults to 100 to match.
*(2) exact default, confirmed non-algorithmic by direct code inspection.*

### Runtime and memory complexity

Per-request cost: O(1) amortized for `S`/`Q` deque operations, O(log n)
for LFU-heap insert/update/remove (`HeapDict`), O(1) for ghost-history
deque operations. Extra memory beyond the physical cache: two ghost
histories of `cache_size // 2` entries each. *(2) exact, by direct
inspection of `DequeDict`/`HeapDict` (both self-documented O(1)/O(log n) in
their own docstrings and self-test blocks).*

### Known upstream limitation: capacity 1

At `cache_size = 1`, `history_size = 1 // 2 = 0`. The official
`addToHistory()` evicts-before-inserting whenever
`len(policy_history) == history_size`, which is true (`0 == 0`) on the
very first history write to an *empty* deque, and `DequeDict.first()` on
an empty deque raises `AttributeError` (`self.head` is `None`). **Confirmed
empirically** against the pinned commit
(`Cacheus(1, 100)` crashes on the second request in a 2-distinct-page
trace). This is an upstream bug/limitation in the official source at this
specific boundary, not an adapter defect. Not patched (would be an
algorithm-changing patch to third-party code); `CacheusPolicy.reset()`
raises an explicit, documented `ValueError` for `capacity < 2` instead of
letting this exception propagate uninterpreted from third-party code.
*(2) exact — an official-source boundary bug, disclosed rather than
silently worked around.* Irrelevant in practice: this repository's
manuscript capacities are 32/64/128, never 1.

### Hyperparameters

| Parameter | Official default | Used here | Source |
|---|---|---|---|
| `initial_weight` | 0.5 | 0.5 (unset → official default) | (2) exact |
| `history_size` | `cache_size // 2` | unset → official default | (2) exact |
| `learning_rate` (initial) | `sqrt(2*ln(2)/cache_size)` | unset → official default | (2) exact |
| `window_size` | 100 (per official `run.py`, for integer cache sizes) | 100 | (2) exact, non-algorithmic |
| RNG seed | 123 (hardcoded in `Cacheus.__init__`) | 123 (not overridable — official source hardcodes it) | (2) exact |

All hyperparameters are used at their **official defaults**; none were
tuned against Wulver trace results (priority 1 of this task's
hyperparameter protocol — literal official defaults). `CacheusConfig`
exposes `initial_weight`/`history_size`/`learning_rate` overrides only for
sensitivity-analysis use, unset (→ official default) in the headline
runner (`scripts/experiments/run_cacheus_comparison.py`).

## Fidelity summary

| Aspect | Classification |
|---|---|
| Expert-combination framework (LeCaR-style Hedge) | (1)/(2) exact |
| SR-LRU (adaptive S/Q partition) | (2) exact |
| CR-LFU (frequency min-heap) | (2) exact |
| Ghost-history delayed feedback | (2) exact |
| Weight-update equation | (2) exact |
| Learning-rate schedule | (2) exact |
| Victim proposal/selection | (2) exact |
| Cold start (uniform prior, no LRU phase) | (2) exact |
| Admission (no bypass) | (2) exact |
| Unit-size/object-slot capacity semantics | (1)/(2) exact — no adaptation needed |
| `oblock`/`ts` identity mapping | (2) exact — no trace-format adapter needed |
| Capacity-1 upstream crash | (2) exact upstream limitation, disclosed and guarded, not patched |
| Hyperparameters | (2) exact official defaults throughout |

**Overall**: this baseline runs the **authors' own, unmodified algorithm**
on this repository's exact trace/capacity/metric protocol, with no
byte-vs-unit-size adaptation needed (a stronger fidelity position than
LRB/3L-Cache/HALP in this repository, all three of which required a
disclosed unit-size adaptation). Reviewer-facing framing: **"CACHEUS
(official source, SR-LRU + CR-LFU, github.com/sylab/cacheus @
`1eec63ce166502be33ddd1f35bc041ed73a24f4d`)."**

## Fairness table (CACHEUS vs. evict_value_v1, LRB, 3L-Cache, HALP, LRU)

| Dimension | evict_value_v1 | LRB | 3L-Cache | HALP | CACHEUS | LRU |
|---|---|---|---|---|---|---|
| Trace identity | 7 Wulver families, 50K req | same | same | same | same | same |
| Simulator | this repo's `run_policy` (native) | native | native | native | **official `Cacheus` class, executed via adapter** | native |
| Capacity semantics | unit-size paging, 32/64/128 | same | same (adapted from byte-cache) | same (adapted from byte-cache) | **same — native to the algorithm, no adaptation** | same |
| Object-size semantics | unit | unit | unit (adapted; official is byte) | unit (adapted; official is byte) | **unit (exact match to official; official has no size concept at all)** | unit |
| Metric | request misses / miss ratio | same | same | same | same | same |
| Training mode | offline, pretrained model | online, batched retraining | online, batched retraining | offline, single frozen split | **online, continuous (official, unmodified)** | none |
| Future information | none | none | none | none (frozen after split) | **none — delayed feedback only from already-observed re-requests** | n/a |
| Hyperparameter source | fixed pretrained config | official defaults where operative | validation-tuned `batch_size` | repository-chosen (no official defaults exist) | **official defaults throughout (all defined and used unmodified)** | n/a |
| Randomization | fixed seed | fixed seed | fixed seed | fixed seed | **fixed at official's hardcoded seed 123 (not repository-controlled)** | n/a |
| Cold start | n/a (pretrained) | plain LRU | plain LRU | plain LRU | **uniform-weight adaptive from t=0 (no LRU phase)** | n/a |
| Fallback | explicit guard variant | LRU-head on stale/degenerate state | heap-exhausted LRU fallback | zero-score tie-break | **none needed; capacity<2 raises explicit error instead** | n/a |
| Official code available | n/a | yes (BSD-2-Clause) | yes (GPL-3.0) | no | **yes, but no LICENSE file (executed externally, never vendored)** | n/a |
| Code executed | this repo's own model | reimplementation | reimplementation | reimplementation | **authors' own unmodified source** | native |

**Non-equivalences that must not be collapsed when combining results:**
CACHEUS's online, continuous, delayed-feedback adaptation is not
equivalent to `evict_value_v1`'s offline pretraining, HALP's frozen
single-split training, or LRB/3L-Cache's batched online retraining. All
five numbers are "a request-miss count under this repository's paging
replay," but CACHEUS is the only one of the five running the *authors' own
code* rather than a reimplementation, and the only one with a
non-repository-controlled RNG seed — both should be stated explicitly
whenever CACHEUS is compared in a table, not silently normalized away.
