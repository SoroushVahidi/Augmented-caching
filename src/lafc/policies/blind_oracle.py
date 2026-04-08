"""
Blind Oracle caching policy (unweighted paging).

Reference
---------
Lykouris, Vassilvitskii.
"Competitive Caching with Machine Learned Advice."
ICML 2018 / JACM 2021.

============================================================
PAPER-TO-CODE IMPLEMENTATION NOTE
============================================================

Setting
-------
Standard paging (unweighted):
- Cache capacity k, all pages unit size.
- All misses have unit cost (cost = 1).
- Each request σ_t arrives with a prediction τ_t (predicted next
  arrival time of σ_t).

Algorithm
---------
The Blind Oracle fully trusts the predictor: on a miss it evicts the
cached page whose *predicted* next arrival is farthest in the future:

    victim = argmax_{q ∈ cache} predicted_next[q]

This is equivalent to running Belady's optimal offline algorithm using
*predicted* next arrivals instead of actual ones.

- When predictions are perfect (τ_t = a_t), Blind Oracle is optimal.
- When predictions are adversarial, Blind Oracle can be arbitrarily bad.

Maintained state
----------------
  _predicted_next : Dict[PageId, float]
      Most recently received predicted next arrival for each page.
      Defaults to math.inf (treat as "never needed again") for pages
      without a prior prediction.

INTERPRETATION NOTE — cost model
---------------------------------
This policy uses unit cost (1.0 per miss) regardless of page weights,
consistent with the unweighted paging setting of Lykouris & Vassilvitskii
2018.  If weighted pages are supplied, their weights are ignored for cost
purposes.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


class BlindOraclePolicy(BasePolicy):
    """Blind Oracle: evict the page with the largest predicted next arrival.

    Fully trusts predictions.  Unit cost per miss (unweighted paging).

    Parameters
    ----------
    There are no configurable parameters; the policy is fully determined
    by the predictions received at each step.
    """

    name: str = "blind_oracle"

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        # INTERPRETATION NOTE: default prediction is math.inf (unknown →
        # treat as "never needed again").  Such pages are candidates for
        # eviction, but they have lower priority than pages with an explicit
        # infinite prediction (both map to math.inf, equal priority; ties
        # broken lexicographically).
        self._predicted_next: Dict[PageId, float] = {}

    # ------------------------------------------------------------------
    # Main algorithm step
    # ------------------------------------------------------------------

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id

        # Always record the latest prediction for the requested page.
        self._predicted_next[pid] = request.predicted_next

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
        """Return the cached page with the largest predicted next arrival.

        Parameters
        ----------
        exclude:
            Page id to exclude from consideration (the page being fetched,
            which is not yet in cache but we guard for robustness).

        Tie-breaking: for equal predicted_next, evict the lexicographically
        largest page_id (deterministic).
        """
        candidates = [q for q in self._cache.current_cache() if q != exclude]
        if not candidates:
            raise RuntimeError(
                f"No eviction candidate found; cache={self._cache.current_cache()}, "
                f"requesting={exclude}"
            )

        def sort_key(q: PageId):
            # Higher predicted_next → evict first; break ties by page_id (desc).
            return (-self._predicted_next.get(q, math.inf), q)

        return min(candidates, key=sort_key)
