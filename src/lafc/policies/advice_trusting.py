"""
Advice-Trusting caching policy.

This policy treats predictions as fully trustworthy and evicts the cached page
whose *predicted* next arrival is farthest in the future (exactly like
Belady's OPT but using *predicted* next arrivals instead of actual ones).

It serves as a consistency-only baseline: it is optimal when predictions are
perfect, but can degrade badly under adversarial predictions.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


class AdviceTrustingPolicy(BasePolicy):
    """Evict the page with the largest predicted next-arrival time."""

    name: str = "advice_trusting"

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        # Stores the most recently received predicted_next for each cached page.
        self._predicted_next: Dict[PageId, float] = {}

    def _choose_eviction_candidate(self) -> PageId:
        """Return the cached page with the largest predicted next arrival."""
        cached = self._cache.current_cache()
        return max(cached, key=lambda q: self._predicted_next.get(q, math.inf))

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id

        # Always update our prediction for the requested page.
        self._predicted_next[pid] = request.predicted_next

        if self.in_cache(pid):
            self._record_hit()
            return CacheEvent(t=request.t, page_id=pid, hit=True, cost=0.0)

        # Cache miss.
        cost = self._pages[pid].weight
        self._record_miss(cost)
        evicted: Optional[PageId] = None

        if self._cache.is_full():
            evicted = self._choose_eviction_candidate()
            self._evict(evicted)

        self._add(pid)

        return CacheEvent(t=request.t, page_id=pid, hit=False, cost=cost, evicted=evicted)
