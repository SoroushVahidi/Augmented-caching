"""
Deterministic BlindOracle + LRU black-box combiner (Baseline 4).

Reference
---------
Alexander Wei.
"Better and Simpler Learning-Augmented Online Caching."
Approximation, Randomization, and Combinatorial Optimization.
Algorithms and Techniques (APPROX/RANDOM 2020).
LIPIcs, Vol. 176, Article 60.

============================================================
PAPER-TO-CODE IMPLEMENTATION NOTE (Wei 2020, Baseline 4)
============================================================
No public code release exists for this paper.  The implementation is
reconstructed from the paper itself.  All ambiguities are documented
with INTERPRETATION NOTE tags.

1. Exact learning-augmented paging model (Wei 2020)
----------------------------------------------------
Standard paging (unweighted caching):
- Cache capacity k; all pages have unit size.
- All misses have unit cost (cost = 1).
- Request sequence σ_1, ..., σ_T.
- Each request σ_t arrives with a prediction τ_t: the predicted next
  arrival time of σ_t after time t.  τ_t = ∞ means "never again."
- Objective: minimise total number of cache misses.

2. Definition of BlindOracle (Wei 2020)
-----------------------------------------
BlindOracle fully trusts the predictor:
    On a cache miss when the cache is full, evict the cached page whose
    *predicted* next arrival τ_q is largest (farthest in the future).

This is equivalent to Belady's optimal offline algorithm run with
predicted arrivals in place of actual arrivals.  When predictions are
perfect (τ_t = a_t for all t), BlindOracle is optimal.  Already
implemented in blind_oracle.py.

3. Prediction error η (Wei 2020)
----------------------------------
    η = Σ_t |τ_t − a_t|

where τ_t is the predicted next arrival and a_t is the actual next
arrival of σ_t at time t.  Both being ∞ contributes 0; one finite and
one infinite contributes ∞.

4. Deterministic black-box combination idea (Wei 2020)
--------------------------------------------------------
Wei 2020 (Section 3, Theorem 1) shows a deterministic online combiner
that achieves a competitive ratio bounded by the *minimum* of the
individual competitive ratios of BlindOracle and LRU (up to constants).

The paper's key algorithmic claim:
    "The optimal deterministic strategy is especially simple: for each
    eviction, follow whichever of BlindOracle and LRU has performed
    better so far."

5. "Follow whichever has performed better" → executable code
-------------------------------------------------------------
INTERPRETATION NOTE 1 — "performed better" definition:
    "Performed better so far" is interpreted as "has incurred fewer
    cache misses so far."  Both BlindOracle and LRU are run as
    INDEPENDENT shadow instances on the same request sequence.  Each
    shadow maintains its own cache state.  Their cumulative miss counts
    are compared to decide the combiner's eviction rule.

    At each request t:
      1.  Read shadow miss counts accumulated from requests 0 .. t−1
          (before processing request t, so the decision is made on
          information strictly prior to time t — fully online).
      2.  If shadow_bo.misses ≤ shadow_lru.misses:
              apply BlindOracle eviction rule to combiner's own cache.
          Else:
              apply LRU eviction rule to combiner's own cache.
      3.  Process request t through BOTH shadows (update their states).

    RATIONALE: "Follow" means applying the eviction RULE of the chosen
    algorithm to the combiner's OWN cache state (not the shadow's cache).
    This avoids the need to synchronise the combiner's cache with a
    shadow's cache, which would require teleporting pages in and out of
    cache — an operation not available in an online algorithm.

6. Hidden assumptions for the combiner
----------------------------------------
INTERPRETATION NOTE 2 — independent shadows:
    The combiner's cache state is independent of both shadow caches.
    The shadows run on the same request sequence but make their own
    independent eviction decisions.  Shadow miss counts are the sole
    channel of information from shadow to combiner.

INTERPRETATION NOTE 3 — eviction rules applied to combiner's cache:
    "BlindOracle rule" applied to the combiner's cache:
        evict argmax_{q ∈ combiner_cache} predicted_next[q]
    using the combiner's OWN predicted_next dict (updated on every
    request received by the combiner).

    "LRU rule" applied to the combiner's cache:
        evict the least-recently-used page in the combiner's cache
    using the combiner's OWN recency order (updated on every request).

INTERPRETATION NOTE 4 — tie-breaking when shadow costs are equal:
    When shadow_bo.misses == shadow_lru.misses the combiner favours
    BlindOracle.  This is an arbitrary but deterministic and
    reproducible choice.  It could be reversed without affecting the
    algorithm's theoretical properties.

INTERPRETATION NOTE 5 — hits and shadow updates:
    Both shadows are updated on every request (hits included), so that
    their cumulative miss counts accurately reflect the full request
    sequence up to the current time.

7. Randomized Equitable-based combination (Wei 2020)
-----------------------------------------------------
Wei 2020 also describes a randomized variant using an H_k-competitive
randomized paging algorithm (e.g. Equitable) in place of LRU.  See
blind_oracle_randomized_combiner.py for a scaffold with TODO markers
explaining what is needed for a faithful implementation.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from lafc.policies.base import BasePolicy
from lafc.policies.blind_oracle import BlindOraclePolicy
from lafc.policies.lru import LRUPolicy
from lafc.types import CacheEvent, Page, PageId, Request


# ---------------------------------------------------------------------------
# Per-step diagnostics
# ---------------------------------------------------------------------------


@dataclass
class CombinerStepLog:
    """Per-step diagnostics for the deterministic BlindOracle + LRU combiner.

    Attributes
    ----------
    t:
        Request index.
    page_id:
        Requested page.
    hit:
        True if the page was in the combiner's cache.
    evicted:
        Page evicted by the combiner (None on a hit or when cache was not full).
    chosen:
        Which sub-algorithm's eviction rule was applied: ``"blind_oracle"``,
        ``"lru"``, or ``None`` (cache hit — no eviction needed).
    bo_misses_before:
        Shadow BlindOracle miss count *before* processing request t.
    lru_misses_before:
        Shadow LRU miss count *before* processing request t.
    """

    t: int
    page_id: PageId
    hit: bool
    evicted: Optional[PageId]
    chosen: Optional[str]
    bo_misses_before: int
    lru_misses_before: int


# ---------------------------------------------------------------------------
# Combiner policy
# ---------------------------------------------------------------------------


class BlindOracleLRUCombiner(BasePolicy):
    """Deterministic BlindOracle + LRU black-box combiner (Wei 2020).

    At each cache miss requiring an eviction, the combiner applies the
    eviction rule of whichever shadow algorithm (BlindOracle or LRU) has
    accumulated fewer cache misses so far.

    Parameters
    ----------
    There are no configurable parameters.  Behaviour is fully determined
    by the predictions received at each step.

    Attributes (exposed for diagnostics)
    ------------------------------------
    step_log()
        Returns the per-step ``CombinerStepLog`` list.
    shadow_bo_misses()
        Current miss count of the shadow BlindOracle instance.
    shadow_lru_misses()
        Current miss count of the shadow LRU instance.
    """

    name: str = "blind_oracle_lru_combiner"

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)

        # Shadow algorithms: independent instances sharing no state with the
        # combiner.  Their sole purpose is to track cumulative miss counts.
        # INTERPRETATION NOTE 2.
        self._shadow_bo: BlindOraclePolicy = BlindOraclePolicy()
        self._shadow_bo.reset(capacity, pages)

        self._shadow_lru: LRUPolicy = LRUPolicy()
        self._shadow_lru.reset(capacity, pages)

        # Combiner's own recency order (for LRU eviction rule).
        # Most-recently-used end = right/end of OrderedDict.
        # INTERPRETATION NOTE 3.
        self._order: collections.OrderedDict[PageId, None] = (
            collections.OrderedDict()
        )

        # Combiner's own predicted next-arrivals (for BlindOracle eviction rule).
        # INTERPRETATION NOTE 3.
        self._predicted_next: Dict[PageId, float] = {}

        # Per-step diagnostics.
        self._step_log: List[CombinerStepLog] = []

    # ------------------------------------------------------------------
    # Main algorithm step
    # ------------------------------------------------------------------

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id

        # Always update combiner's own predicted_next for the requested page.
        # This mirrors BlindOraclePolicy's behaviour (see blind_oracle.py).
        self._predicted_next[pid] = request.predicted_next

        # Read shadow miss counts BEFORE processing this request.
        # This ensures decisions at time t are based on information from
        # times 0 .. t−1 only — the algorithm is fully online.
        # INTERPRETATION NOTE 1.
        bo_misses_before: int = self._shadow_bo._misses
        lru_misses_before: int = self._shadow_lru._misses

        # ----------------------------------------------------------------
        # Cache hit
        # ----------------------------------------------------------------
        if self.in_cache(pid):
            # Update combiner's LRU recency order (for future evictions).
            self._order.move_to_end(pid)
            self._record_hit()

            # Both shadows are updated on every request (hits included).
            # INTERPRETATION NOTE 5.
            self._shadow_bo.on_request(request)
            self._shadow_lru.on_request(request)

            self._step_log.append(
                CombinerStepLog(
                    t=request.t,
                    page_id=pid,
                    hit=True,
                    evicted=None,
                    chosen=None,
                    bo_misses_before=bo_misses_before,
                    lru_misses_before=lru_misses_before,
                )
            )
            return CacheEvent(t=request.t, page_id=pid, hit=True, cost=0.0)

        # ----------------------------------------------------------------
        # Cache miss — unit cost (unweighted paging).
        # ----------------------------------------------------------------
        self._record_miss(1.0)
        evicted: Optional[PageId] = None
        chosen: Optional[str] = None

        if self._cache.is_full():
            # Choose sub-algorithm based on which shadow has fewer misses.
            # Tie broken in favour of BlindOracle.  INTERPRETATION NOTE 4.
            if bo_misses_before <= lru_misses_before:
                chosen = "blind_oracle"
                evicted = self._evict_by_blind_oracle_rule(exclude=pid)
            else:
                chosen = "lru"
                evicted = self._evict_by_lru_rule()

            # Remove evicted page from combiner's bookkeeping.
            self._evict(evicted)
            self._order.pop(evicted, None)

        # Fetch the requested page.
        self._add(pid)
        self._order[pid] = None  # insert at most-recently-used end

        # Update both shadows AFTER the combiner's eviction decision.
        # Their costs thus reflect requests 0 .. t−1 at the moment of
        # comparison (step 1 above) and 0 .. t after the update.
        # INTERPRETATION NOTE 1.
        self._shadow_bo.on_request(request)
        self._shadow_lru.on_request(request)

        self._step_log.append(
            CombinerStepLog(
                t=request.t,
                page_id=pid,
                hit=False,
                evicted=evicted,
                chosen=chosen,
                bo_misses_before=bo_misses_before,
                lru_misses_before=lru_misses_before,
            )
        )
        return CacheEvent(
            t=request.t, page_id=pid, hit=False, cost=1.0, evicted=evicted
        )

    # ------------------------------------------------------------------
    # Eviction rules (applied to COMBINER's own cache, not a shadow's)
    # INTERPRETATION NOTE 3.
    # ------------------------------------------------------------------

    def _evict_by_blind_oracle_rule(self, exclude: PageId) -> PageId:
        """Evict the cached page with the largest predicted_next.

        Uses the combiner's own predicted_next dict.

        Parameters
        ----------
        exclude:
            Page being fetched (not yet in cache; excluded for safety).

        Tie-breaking: lexicographically largest page_id (deterministic).
        """
        candidates = [q for q in self._cache.current_cache() if q != exclude]
        if not candidates:
            raise RuntimeError(
                f"BlindOracle rule: no eviction candidate; "
                f"cache={self._cache.current_cache()}, requesting={exclude}"
            )

        def sort_key(q: PageId):
            # Larger predicted_next → evict first; break ties by page_id.
            return (-self._predicted_next.get(q, math.inf), q)

        return min(candidates, key=sort_key)

    def _evict_by_lru_rule(self) -> PageId:
        """Evict the least-recently-used page from the combiner's cache.

        Uses the combiner's own LRU order (the front of the OrderedDict).
        """
        if not self._order:
            raise RuntimeError("LRU rule: order is empty; cannot evict.")
        # Front = least-recently-used.
        return next(iter(self._order))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def step_log(self) -> List[CombinerStepLog]:
        """Return a copy of the per-step decision log."""
        return list(self._step_log)

    def shadow_bo_misses(self) -> int:
        """Current miss count of the shadow BlindOracle instance."""
        return self._shadow_bo._misses

    def shadow_lru_misses(self) -> int:
        """Current miss count of the shadow LRU instance."""
        return self._shadow_lru._misses
