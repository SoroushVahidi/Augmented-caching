"""
Weighted-LRU caching policy.

A cost-aware variant of LRU: on a cache miss, when the cache is full, evict
the cached page with the **smallest weight** (cheapest to re-fetch).
Ties in weight are broken by recency — the least-recently-used among the
minimum-weight pages is evicted.

This is the natural "weighted" extension of LRU that takes page weights into
account without using predictions.
"""

from __future__ import annotations

import collections
from typing import Dict, Optional

from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


class WeightedLRUPolicy(BasePolicy):
    """Weighted-LRU: evict the cheapest page on a fault."""

    name: str = "weighted_lru"

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        # Track recency order for tie-breaking.
        self._order: collections.OrderedDict[PageId, None] = collections.OrderedDict()

    def _choose_eviction_candidate(self) -> PageId:
        """Return the page id to evict.

        Strategy: ``argmin_{q in cache} w_q``, breaking ties by LRU order
        (least-recently-used among min-weight pages).
        """
        cached = list(self._order.keys())  # oldest → newest
        min_weight = min(self._pages[q].weight for q in cached)
        # Among pages with minimum weight, pick the least-recently-used.
        for q in cached:  # oldest first
            if self._pages[q].weight == min_weight:
                return q
        # Unreachable, but keeps mypy happy.
        raise RuntimeError("No eviction candidate found")  # pragma: no cover

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id
        evicted: Optional[PageId] = None

        if self.in_cache(pid):
            self._order.move_to_end(pid)
            self._record_hit()
            return CacheEvent(t=request.t, page_id=pid, hit=True, cost=0.0)

        # Cache miss.
        cost = self._pages[pid].weight
        self._record_miss(cost)

        if self._cache.is_full():
            evicted = self._choose_eviction_candidate()
            self._order.pop(evicted)
            self._evict(evicted)

        self._add(pid)
        self._order[pid] = None

        return CacheEvent(t=request.t, page_id=pid, hit=False, cost=cost, evicted=evicted)
