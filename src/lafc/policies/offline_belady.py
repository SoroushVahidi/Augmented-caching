"""
Offline Belady optimal caching policy.

This is the classical Belady (1966) "furthest in the future" algorithm,
using *actual* next arrivals derived from the trace.  It is offline (it
uses ground-truth future information) and serves only as an OPT baseline
for evaluation on small traces.

DO NOT use this policy to make production eviction decisions — it cheats
by consulting actual_next, which is derived from the future of the trace.

Reference
---------
László A. Bélády.
"A Study of Replacement Algorithms for a Virtual Storage Computer."
IBM Systems Journal, 5(2), 1966.

============================================================
PAPER-TO-CODE IMPLEMENTATION NOTE
============================================================

Setting
-------
Standard paging (unweighted), cache capacity k, unit cost per miss.

Algorithm
---------
On a cache miss when the cache is full:
    evict the page q in cache whose actual next arrival a_q is
    largest (farthest in the future).  If a_q = ∞ for multiple pages,
    any of them is valid; we break ties by page_id lexicographically.

Maintained state
----------------
  _actual_next : Dict[PageId, float]
      For each page, the most recent actual_next value received
      (i.e. the actual next arrival time after the last time this
      page was requested).

INTERPRETATION NOTE — why this is correct
-------------------------------------------
The _actual_next entry for a cached page q is set when q was last
requested at some time t' ≤ t.  Because q has not been requested again
between t' and t (it would have been requested and the entry updated),
the stored value is the actual next arrival of q after t', which is also
the actual next arrival after any t in (t', actual_next[q]).  Thus
_actual_next[q] correctly represents when q will next be needed.

INTERPRETATION NOTE — cost model
----------------------------------
Unit cost per miss, consistent with the unweighted paging setting.
Page weights from the trace are ignored.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


class OfflineBeladyPolicy(BasePolicy):
    """Belady's optimal offline 'furthest in the future' policy.

    Uses ``request.actual_next`` (ground-truth, computed from the trace) for
    eviction decisions.  This is **not** an online algorithm.

    Parameters
    ----------
    There are no configurable parameters.
    """

    name: str = "offline_belady"

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        # Ground-truth next-arrival times for cached pages.
        self._actual_next: Dict[PageId, float] = {}

    # ------------------------------------------------------------------
    # Main algorithm step
    # ------------------------------------------------------------------

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id

        # Record ground-truth next arrival for the requested page.
        self._actual_next[pid] = request.actual_next

        if self.in_cache(pid):
            self._record_hit()
            return CacheEvent(t=request.t, page_id=pid, hit=True, cost=0.0)

        # Cache miss — unit cost.
        self._record_miss(1.0)
        evicted: Optional[PageId] = None

        if self._cache.is_full():
            evicted = self._choose_victim(exclude=pid)
            self._evict(evicted)

        self._add(pid)

        return CacheEvent(
            t=request.t, page_id=pid, hit=False, cost=1.0, evicted=evicted
        )

    # ------------------------------------------------------------------
    # Eviction logic
    # ------------------------------------------------------------------

    def _choose_victim(self, exclude: PageId) -> PageId:
        """Return the cached page with the largest actual_next.

        Parameters
        ----------
        exclude:
            The page being fetched (not yet in cache; excluded for safety).

        Tie-breaking: lexicographically largest page_id (deterministic).
        """
        candidates = [q for q in self._cache.current_cache() if q != exclude]
        if not candidates:
            raise RuntimeError(
                f"Belady: no eviction candidate; "
                f"cache={self._cache.current_cache()}, requesting={exclude}"
            )

        def sort_key(q: PageId):
            # Larger actual_next → evict first; break ties by page_id.
            return (-self._actual_next.get(q, math.inf), q)

        return min(candidates, key=sort_key)
