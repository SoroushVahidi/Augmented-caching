# SIEVE source verification (2026-06-19)

Zero-compute research pass. No code was written in producing this report —
this is the required pre-implementation verification step. Per instruction,
**implementation does not proceed until this report clearly states the
algorithm semantics**, which it now does (§7-8 below).

## 1. Paper title

"SIEVE is Simpler than LRU: an Efficient Turn-Key Eviction Algorithm for
Web Caches"

## 2. Authors

Yazhuo Zhang (Emory University), Juncheng Yang (Carnegie Mellon University,
corresponding author), Yao Yue (Pelikan Foundation), Ymir Vigfusson (Emory
University & Keystrike), K. V. Rashmi (Carnegie Mellon University).

## 3. Venue/year

21st USENIX Symposium on Networked Systems Design and Implementation
(NSDI '24), April 16-18, 2024, Santa Clara, CA, USA. Pages 1229-1246.
ISBN 978-1-939133-39-7. Won the Community Award at NSDI'24.

## 4. URL/DOI/paper link

- Official USENIX proceedings PDF (downloaded and read directly for this
  report): `https://www.usenix.org/system/files/nsdi24-zhang-yazhuo.pdf`
- USENIX presentation/abstract page:
  `https://www.usenix.org/conference/nsdi24/presentation/zhang-yazhuo`
- Author mirror (not used as primary source, listed for completeness):
  `https://junchengyang.com/publication/nsdi24-SIEVE.pdf`
- No DOI is listed on the USENIX proceedings page (USENIX NSDI papers
  typically do not have a separate DOI beyond the proceedings URL above).

## 5. Whether official code exists

**Yes.** Official author repository:
`https://github.com/cacheMon/NSDI24-SIEVE` (a snapshot/companion repo for
the paper, containing traces, prototypes in 5 languages, and a snapshot of
the `libCacheSim` simulator). The actual reference simulator implementation
lives in the standalone `libCacheSim` project
(`https://github.com/1a1a11a/libCacheSim`,
`libCacheSim/cache/eviction/Sieve.c`), which was fetched directly for this
report and **independently confirms** the paper's Algorithm 1 (see §6-7
below — the C implementation's `freq` field plays exactly the role of
Algorithm 1's `visited` bit, with identical control flow). This
independent code/paper cross-check is the strongest possible verification
available without running the official simulator itself.

## 6. Exact pseudocode / algorithm semantics

Quoted directly from **Algorithm 1 SIEVE** in the paper (§3.1, p.1232):

```
Algorithm 1 SIEVE
Input: The request x, doubly-linked queue T, cache size C, hand p
 1: if x is in T then                          ▷ Cache Hit
 2:   x.visited ← 1
 3: else
 4:   if |T| = C then                          ▷ Cache Miss, Cache Full
 5:     o ← p
 6:     if o is NULL then
 7:       o ← tail of T
 8:     while o.visited = 1 do
 9:       o.visited ← 0
10:       o ← o.prev
11:       if o is NULL then
12:         o ← tail of T
13:     p ← o.prev
14:     Discard o in T                          ▷ Eviction
15:   Insert x in the head of T                 ▷ Insertion
16:   x.visited ← 0
```

**Prose description (§3.1, paraphrasing the paper directly):**

- **Data structure**: exactly one FIFO doubly-linked queue `T`, plus one
  pointer called "hand" (`p`). Each object in the queue carries one bit:
  visited/non-visited. The queue is depicted as `|head(new)--[hand]--tail(old)|`
  (Figure 1) — the head holds the most recently inserted objects, the tail
  holds the oldest.
- **Cache hit**: sets `x.visited ← 1`. **The object is not moved** — this
  is the key difference from LRU (no eager promotion to head on hit).
- **Cache miss, cache full**: start scanning at the hand `p` (or at the
  tail of `T` if `p` is `NULL`, i.e. on the very first eviction or after
  the hand has wrapped past the head). Walk via `.prev` (which moves from
  tail-side toward head-side, since `.prev` is defined relative to
  insertion order with new items at the head). For each object visited
  during the scan with `visited = 1`: **reset its visited bit to 0** (this
  is the "give a second chance" step) and advance to `.prev`. If the walk
  passes the head (`.prev` is `NULL`), wrap around to the tail of `T` and
  continue. The scan stops at the **first object found with
  `visited = 0`** — that object is evicted.
- **Hand update after eviction**: `p ← o.prev`, i.e. the hand is left
  pointing at the position immediately preceding (toward the head from)
  the just-evicted object, so the **next** eviction resumes scanning from
  there rather than restarting at the tail every time.
- **Insertion**: every newly admitted object goes to the **head** of `T`
  with `visited` initialized to **0** (`x.visited ← 0`, line 16). The hand
  itself is **not** moved or reset on insertion — only the scan above
  (triggered by the next full-cache miss) advances it.
- **Critical distinction from CLOCK/Second-Chance/FIFO-Reinsertion**
  (paper's own framing, §3.1, p.1232): those algorithms keep the hand
  fixed at the tail and move *retained* (survived) objects to the head on
  a second-chance pass. SIEVE instead **moves the hand itself** toward the
  head over successive evictions, and **leaves retained objects in their
  original queue position** — "the new objects and the retained objects
  are not mixed together." This is the one substantive behavioral
  difference between SIEVE and a textbook FIFO-Reinsertion/CLOCK variant
  built on the same queue+bit primitives.

**Independent code cross-check** (from the official `libCacheSim`
`Sieve.c`, fetched directly for this report): the real C implementation
uses a `freq` field (values 0/1) as the visited bit, sets
`cache_obj->sieve.freq = 1` on a hit, initializes `obj->sieve.freq = 0` on
insertion (`prepend_obj_to_head`), starts the hand at `q_tail` if `NULL`,
walks backward via `obj->queue.prev` (wrapping to `q_tail` if `NULL`)
while `freq > 0` (decrementing/clearing it as it passes), evicts the first
object with `freq == 0`, and sets `pointer = obj->queue.prev` afterward.
This **exactly** matches Algorithm 1 line-for-line — no discrepancy found
between the paper's pseudocode and the official reference implementation.

## 7. Ambiguity when adapting from object-size web caching to unweighted paging

**None of substance.** This is the cleanest possible case for adaptation:

- Algorithm 1 itself is already defined purely in terms of **objects**,
  not bytes — the only place a size/capacity unit appears is the
  "cache full" check, `|T| = C`, which is an **object count** comparison,
  not a byte-budget comparison. The paper's broader system (web/CDN
  caches) does track byte sizes for the *admission/space-accounting*
  layer outside Algorithm 1, but the eviction algorithm itself evicts and
  inserts exactly **one object at a time**, regardless of object size.
- This repo's simulator is **already unweighted paging**: every resident
  item is one page, capacity is a page count, and "cache full" is already
  `|resident set| = capacity`. This is a literal match to Algorithm 1's
  own `|T| = C` condition — no reinterpretation, no per-object weighting,
  no proportional/size-adjusted variant is needed.
- The only thing genuinely adapted (not reinterpreted) is vocabulary:
  "object" → "page", "web cache request" → "page request." The algorithm's
  control flow (hand position, visited-bit scan/clear, head-insertion,
  hand update after eviction) transfers unchanged.

## 8. Final implementation decision

**Implement SIEVE faithfully per Algorithm 1, with the following exact
mapping to this repo's `BasePolicy` interface** (see
`reports/kbs_sieve_implementation_report.md` for the concrete code):

- Maintain residents in insertion order using a structure that supports
  O(1) append-at-head and O(1) remove-from-middle (a `dict`-backed doubly
  linked list, or equivalently Python's `dict` insertion order plus a
  separate `visited: Dict[key, bool]` map and an explicit `hand` cursor —
  exact data-structure choice is an implementation detail of step 4, not a
  semantic question, since this report's job is to pin down semantics, not
  code).
- One `visited` bit per resident key, defaulting to `False` (0) at
  insertion (Alg. 1 line 16).
- One `hand` cursor, initialized to `None`/unset; on a full-cache miss,
  resume from the hand (or start at the tail/oldest end if unset/wrapped),
  scan toward the head/newest end via the "previous-in-insertion-order"
  direction, clearing `visited` to `False` for every `True` object passed,
  wrapping from past-the-head back to the tail if the scan runs off the
  newest end, and evicting the first object found with `visited = False`.
- After eviction, set the hand to the evicted object's predecessor
  (toward the head), exactly as `p ← o.prev` in line 13.
- On a cache hit, set `visited = True` for the accessed key and do
  **nothing else** — no reordering, no movement (this is the one rule
  most likely to be implemented wrong if copying LRU-style code, since
  LRU's hit path moves the object to the head; SIEVE's hit path must not).

No ambiguity remains that would block implementation. Proceeding to Step 3
(repo policy-interface inspection) and Step 4 (implementation).
