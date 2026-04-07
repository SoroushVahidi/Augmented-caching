"""
Least-Recently-Used (LRU) caching policy.

A standard LRU implementation backed by :class:`collections.OrderedDict`.
On a cache hit the page is moved to the "most recent" end of the dict;
on a miss the "least recent" end is evicted when the cache is full.

This is a deterministic, oblivious baseline that ignores predictions.
"""

from __future__ import annotations

import collections
from typing import Dict, Optional

from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


class LRUPolicy(BasePolicy):
    """Least-Recently-Used eviction policy."""

    name: str = "lru"

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        # OrderedDict used as an ordered set: keys are page ids,
        # values are None.  Most-recently-used end = end of dict.
        self._order: collections.OrderedDict[PageId, None] = collections.OrderedDict()

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id
        evicted: Optional[PageId] = None

        if self.in_cache(pid):
            # Cache hit: move to most-recently-used end.
            self._order.move_to_end(pid)
            self._record_hit()
            return CacheEvent(t=request.t, page_id=pid, hit=True, cost=0.0)

        # Cache miss.
        cost = self._pages[pid].weight
        self._record_miss(cost)

        if self._cache.is_full():
            # Evict the least-recently-used page (front of OrderedDict).
            evicted, _ = self._order.popitem(last=False)
            self._evict(evicted)

        # Fetch the requested page.
        self._add(pid)
        self._order[pid] = None  # insert at most-recently-used end

        return CacheEvent(t=request.t, page_id=pid, hit=False, cost=cost, evicted=evicted)
