"""
Least-Frequently-Used (LFU) caching policy.

This is a deterministic object-count baseline with complete admission. Every
miss inserts the requested page; when the cache is full, the victim is the
resident page with the lowest access frequency. Ties are broken by oldest last
touch, then by oldest insertion order.
"""

from __future__ import annotations

from typing import Dict, Optional

from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


class LFUPolicy(BasePolicy):
    """Least-Frequently-Used eviction policy with deterministic tie-breaking."""

    name: str = "lfu"

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        self._frequency: Dict[PageId, int] = {}
        self._last_touch: Dict[PageId, int] = {}
        self._insert_order: Dict[PageId, int] = {}
        self._next_insert_order = 0

    def _touch(self, page_id: PageId, t: int) -> None:
        self._frequency[page_id] = self._frequency.get(page_id, 0) + 1
        self._last_touch[page_id] = t

    def _choose_victim(self) -> PageId:
        if not self._frequency:
            raise ValueError("Cannot choose LFU victim from an empty cache")
        return min(
            self._frequency,
            key=lambda pid: (
                self._frequency[pid],
                self._last_touch[pid],
                self._insert_order[pid],
            ),
        )

    def _forget(self, page_id: PageId) -> None:
        del self._frequency[page_id]
        del self._last_touch[page_id]
        del self._insert_order[page_id]

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id
        evicted: Optional[PageId] = None

        if self.in_cache(pid):
            self._touch(pid, request.t)
            self._record_hit()
            return CacheEvent(t=request.t, page_id=pid, hit=True, cost=0.0)

        cost = self._pages[pid].weight
        self._record_miss(cost)

        if self._cache.is_full():
            evicted = self._choose_victim()
            self._evict(evicted)
            self._forget(evicted)

        self._add(pid)
        self._frequency[pid] = 0
        self._insert_order[pid] = self._next_insert_order
        self._next_insert_order += 1
        self._touch(pid, request.t)

        return CacheEvent(t=request.t, page_id=pid, hit=False, cost=cost, evicted=evicted)
