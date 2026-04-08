"""
Predictive Marker caching policy.

The Predictive Marker algorithm augments the classical Marker algorithm with
next-arrival predictions: instead of using LRU order to decide which unmarked
page to evict, it evicts the unmarked page whose *predicted* next arrival is
farthest in the future.

Reference:
    Lykouris, Vassilvitskii.
    "Competitive Caching with Machine Learned Advice."
    ICML 2018.

The algorithm achieves:
- **Consistency**: O(1+η/OPT)-competitive when predictions are accurate.
- **Robustness**: O(log k)-competitive even under adversarial predictions
  (the unmarked-page restriction ensures no worse than standard Marker).

Setting: unweighted paging (unit costs), cache of k pages.

Prediction interface: ``request.predicted_next`` (next arrival time of the
requested page after time t; ``math.inf`` means "never again").

Algorithm
----------
Identical to :class:`~lafc.policies.marker.MarkerPolicy` except the
eviction rule:

  - Evict the unmarked cached page with the **largest** ``predicted_next``.
  - If all pages are marked (phase boundary), unmark all and then evict
    the page with the largest ``predicted_next`` among all cached pages.

Paper-to-code mapping
----------------------
| Concept              | Code location                           |
|----------------------|-----------------------------------------|
| Phase boundary       | ``len(unmarked) == 0`` branch           |
| Prediction-guided    | ``_choose_eviction_candidate()``        |
| Predicted next store | ``_predicted_next: Dict[PageId, float]``|
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set

from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


class PredictiveMarkerPolicy(BasePolicy):
    """Prediction-guided Marker: evict the unmarked page with max predicted_next.

    Consistent under good predictions; O(log k)-robust like standard Marker.

    Reference: Lykouris & Vassilvitskii, ICML 2018.
    """

    name: str = "predictive_marker"

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        # Most recently received predicted next-arrival for each page.
        # Initialised to math.inf: unknown pages are treated as
        # "predicted to stay away a long time" → first eviction candidates.
        #
        # INTERPRETATION NOTE: same default as BlindOraclePolicy and
        # LAWeightedPagingDeterministic.  See docs/baselines.md.
        self._predicted_next: Dict[PageId, float] = {
            pid: math.inf for pid in pages
        }
        # Pages marked in the current phase.
        self._marked: Set[PageId] = set()
        # Diagnostic: number of completed phases.
        self._phase_count: int = 0

    # ------------------------------------------------------------------
    # Eviction rule
    # ------------------------------------------------------------------

    def _choose_eviction_candidate(self, exclude: PageId) -> PageId:
        """Return the unmarked cached page with the largest predicted_next.

        If no unmarked page exists, start a new phase (unmark all) and then
        return the page with the largest predicted_next among all cached pages.

        Parameters
        ----------
        exclude:
            The page being fetched (not yet in cache); exclude it from
            consideration to be safe.
        """
        cached = [q for q in self._cache.current_cache() if q != exclude]
        unmarked: List[PageId] = [q for q in cached if q not in self._marked]

        if not unmarked:
            # Phase boundary: unmark all.
            self._marked = set()
            self._phase_count += 1
            unmarked = cached  # now all are unmarked

        if not unmarked:
            raise RuntimeError("No eviction candidate found.")

        # Evict page with max predicted_next; break ties by page_id.
        return max(unmarked, key=lambda q: (self._predicted_next.get(q, math.inf), q))

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id

        # Update prediction before any decision.
        self._predicted_next[pid] = request.predicted_next

        if self.in_cache(pid):
            self._marked.add(pid)
            self._record_hit()
            return CacheEvent(t=request.t, page_id=pid, hit=True, cost=0.0)

        # Cache miss.
        cost = self._pages[pid].weight
        self._record_miss(cost)
        evicted: Optional[PageId] = None

        if self._cache.is_full():
            evicted = self._choose_eviction_candidate(exclude=pid)
            self._evict(evicted)

        self._add(pid)
        self._marked.add(pid)

        return CacheEvent(t=request.t, page_id=pid, hit=False, cost=cost, evicted=evicted)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def phase_count(self) -> int:
        """Number of phases completed so far."""
        return self._phase_count

    def marked_snapshot(self) -> frozenset:
        """Return a snapshot of the currently marked page ids."""
        return frozenset(self._marked)
