"""
Deterministic Learning-Augmented Weighted Paging.

Reference
---------
Bansal, Coester, Kumar, Purohit, Vee.
"Learning-Augmented Weighted Paging."
SODA 2022.

============================================================
PAPER-TO-CODE IMPLEMENTATION NOTE
============================================================

Setting (Section 2 of the paper)
----------------------------------
- Cache capacity k (unit-size pages, at most k pages at a time).
- Page p has fetch cost (weight) w_p > 0.
- At time t, request σ_t arrives; we also see prediction τ_t for the
  next arrival time of σ_t.
- On a fault for page p, we pay w_p.

Weight classes (Section 3 / Appendix)
---------------------------------------
The paper groups pages into weight classes.  Two natural choices:
  (a) Exact classes: pages with identical weight share a class.
  (b) Rounded classes: round each w_p to the nearest power of 2,
      giving O(log W) classes where W = max_weight / min_weight.

This implementation uses exact weight classes by default (choice a),
with an optional helper ``round_weight_to_power_of_2`` for choice (b).

Deterministic algorithm — core idea (Theorem 1)
-------------------------------------------------
The paper's deterministic algorithm achieves:
  - O(1)-competitive when predictions are perfect (consistent), and
  - O(log k)-competitive against any adversary (robust).

The key algorithmic idea is a *prediction-guided eviction ordering*
that generalises Belady's OPT to the weighted setting:

  Within a single weight class (all pages have equal weight w):
    Evict the page whose predicted next arrival τ_q is largest —
    exactly Belady-with-predictions.  This yields O(1) consistency
    within the class.

  Across weight classes:
    The paper uses a water-filling / primal-dual framework where
    eviction "pressure" is distributed across classes.  The key
    structural property is:
      Lighter (cheaper) pages are evicted more readily for the same
      predicted next-arrival value, because re-fetching them costs less.
    This is captured by normalising by weight: the effective eviction
    priority of cached page q is

        eviction_score(q)  =  predicted_next[q] / w_q            (*)

    Pages with high eviction_score are evicted first.

INTERPRETATION NOTE
--------------------
The paper's Algorithm 1 (as best we can determine from the published
version) uses a more involved primal-dual / water-filling mechanism.
Formula (*) is our interpretation of the *induced ordering* over cached
pages that is consistent with the paper's theoretical guarantees.

Specifically:
  - Setting all weights equal recovers standard Belady-with-predictions
    (τ_q / w = constant * τ_q, so the ranking is identical).
  - Dividing by w_q means that among two pages with the same τ value,
    the cheaper one is evicted — matching the intuition that lighter
    pages should be replaced first.
  - This is consistent with the "class-structured water filling" idea
    in the paper where eviction rate from class i is ∝ 1/w_i.

Any deviation from the paper's exact mechanism is documented with the
tag "INTERPRETATION NOTE" in the code below.

Maintained state
-----------------
  predicted_next : Dict[PageId, float]
      Most recently received predicted next-arrival for every page we
      have seen a prediction for.  Initialised to math.inf (treat as
      "never needed" → first-to-evict candidate if it is also cheap).

  weight_classes : Dict[float, WeightClass]
      One entry per distinct weight value, grouping page ids.

  _cache : CacheState
      Which pages are currently in cache (from BasePolicy).

Update logic
-------------
For every request at time t for page p with prediction τ_t:
  1. Update predicted_next[p] ← τ_t.
  2. If p is in cache: hit, no further action.
  3. If p is not in cache (miss):
       a. Pay cost w_p.
       b. If cache is full:
            evict q* = argmax_{q in cache, q ≠ p} eviction_score(q)
            where eviction_score(q) = predicted_next[q] / w_q.
       c. Add p to cache.

============================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


# ---------------------------------------------------------------------------
# Weight class bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class WeightClass:
    """A group of pages sharing the same fetch cost.

    Attributes
    ----------
    weight:
        The common fetch cost of all pages in this class.
    page_ids:
        Set of page ids belonging to this class (all pages, not just cached).
    """

    weight: float
    page_ids: Set[PageId] = field(default_factory=set)


def round_weight_to_power_of_2(weight: float) -> float:
    """Round *weight* up to the nearest power of 2.

    Use this helper to create O(log W) weight classes for experiments that
    require a logarithmic number of classes (as discussed in the paper).

    Examples
    --------
    >>> round_weight_to_power_of_2(3.0)
    4.0
    >>> round_weight_to_power_of_2(4.0)
    4.0
    >>> round_weight_to_power_of_2(5.0)
    8.0
    """
    if weight <= 0:
        raise ValueError(f"weight must be > 0, got {weight}")
    p = 1.0
    while p < weight:
        p *= 2.0
    return p


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------


class LAWeightedPagingDeterministic(BasePolicy):
    """Deterministic Learning-Augmented Weighted Paging.

    Based on:
        Bansal, Coester, Kumar, Purohit, Vee.
        "Learning-Augmented Weighted Paging."  SODA 2022.

    Eviction rule (see module-level docstring for full derivation):
        On a miss for page p, evict cached page q* that maximises
            predicted_next[q] / w_q
        i.e. farthest-in-future predicted arrival, normalised by weight.

    Parameters
    ----------
    round_weights:
        If True, round page weights to the nearest power of 2 before
        grouping into weight classes.  This creates O(log W) classes
        instead of one class per distinct weight.  Useful for experiments
        matching the paper's O(log ℓ) class count.
    """

    name: str = "la_weighted_paging_deterministic"

    def __init__(self, round_weights: bool = False) -> None:
        self._round_weights = round_weights

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)

        # predicted_next[p]: last received prediction for page p.
        # Initialised to math.inf — "we don't know when p will be needed;
        # assume far future" — conservative default that avoids unnecessary
        # evictions of pages we have no information about.
        #
        # INTERPRETATION NOTE: The paper does not specify a default for pages
        # whose prediction has not yet been observed.  Using math.inf means
        # such pages are treated as low-priority to keep, which matches the
        # spirit of "evict what you won't need for longest."
        self._predicted_next: Dict[PageId, float] = {
            pid: math.inf for pid in pages
        }

        # Build weight classes: group page ids by (optionally rounded) weight.
        self._weight_classes: Dict[float, WeightClass] = {}
        for pid, page in pages.items():
            w = (
                round_weight_to_power_of_2(page.weight)
                if self._round_weights
                else page.weight
            )
            if w not in self._weight_classes:
                self._weight_classes[w] = WeightClass(weight=w)
            self._weight_classes[w].page_ids.add(pid)

    # ------------------------------------------------------------------
    # Main algorithm step
    # ------------------------------------------------------------------

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id

        # Step 1 — update prediction for the requested page.
        # This must happen regardless of hit/miss, because the prediction
        # tells us about the page's *next* arrival after this one.
        self._predicted_next[pid] = request.predicted_next

        # Step 2 — cache hit: nothing else to do.
        if self.in_cache(pid):
            self._record_hit()
            return CacheEvent(t=request.t, page_id=pid, hit=True, cost=0.0)

        # Step 3 — cache miss.
        cost = self._pages[pid].weight
        self._record_miss(cost)
        evicted: Optional[PageId] = None

        if self._cache.is_full():
            evicted = self._choose_eviction_candidate(requesting_page=pid)
            self._evict(evicted)

        self._add(pid)

        return CacheEvent(
            t=request.t, page_id=pid, hit=False, cost=cost, evicted=evicted
        )

    # ------------------------------------------------------------------
    # Eviction logic
    # ------------------------------------------------------------------

    def _eviction_score(self, page_id: PageId) -> float:
        """Compute the eviction priority score for a cached page.

        Higher score → evict first.

        Formula (see module docstring for derivation):
            score(q) = predicted_next[q] / w_q

        Interpretation: a page that is predicted to be needed far in the
        future AND is cheap to re-fetch should be evicted before a page
        that is expensive and needed soon.

        INTERPRETATION NOTE: Division by weight is our interpretation of
        the cross-class water-filling ordering.  Within a single weight
        class the denominator is constant, so the ranking equals Belady-
        with-predictions (consistent).
        """
        w = self._pages[page_id].weight
        tau = self._predicted_next.get(page_id, math.inf)
        # Guard against weight = 0 (blocked by Page.__post_init__).
        return tau / w

    def _choose_eviction_candidate(self, requesting_page: PageId) -> PageId:
        """Return the cached page with the highest eviction score.

        We exclude *requesting_page* from consideration because it is about
        to be fetched (and is not yet in cache at this point, but we guard
        anyway for robustness).

        Tie-breaking: for equal eviction scores, prefer the lighter page
        (cheaper to re-fetch), then the page encountered earliest in the
        cache insertion order (deterministic).
        """
        candidates = [
            q for q in self._cache.current_cache() if q != requesting_page
        ]
        if not candidates:
            raise RuntimeError(
                f"No eviction candidate found; cache={self._cache.current_cache()}, "
                f"requesting={requesting_page}"
            )

        # Primary key: eviction_score (descending).
        # Secondary key: weight (ascending — prefer cheaper to evict on tie).
        # Tertiary key: page_id string (ascending — deterministic).
        def sort_key(q: PageId):
            return (-self._eviction_score(q), self._pages[q].weight, q)

        return min(candidates, key=sort_key)

    # ------------------------------------------------------------------
    # Diagnostics / debugging
    # ------------------------------------------------------------------

    def weight_classes(self) -> Dict[float, WeightClass]:
        """Return the weight-class structure (for inspection / tests)."""
        return dict(self._weight_classes)

    def predicted_next_snapshot(self) -> Dict[PageId, float]:
        """Return a copy of the current predicted_next mapping."""
        return dict(self._predicted_next)
