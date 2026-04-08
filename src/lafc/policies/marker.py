"""
Deterministic Marker (LRU-Marker) caching policy.

The Marker algorithm is a classical online caching algorithm that achieves
an O(H_k) = O(log k) competitive ratio in the unweighted setting.  It is
used as the *robustness sub-routine* inside TRUST&DOUBT and is also a
standalone baseline.

Reference (original randomized Marker):
    Fiat, Karp, Luby, McGeoch, Sleator, Young.
    "Competitive Paging Algorithms."  Journal of Algorithms, 1991.

Deterministic variant (LRU-Marker):
    Use LRU order among unmarked pages for a deterministic implementation.
    This preserves the Θ(log k) competitive-ratio guarantee in the worst case.

Setting: unweighted paging (unit costs), cache of k pages.

Algorithm
----------
Operates in **phases**.  At the start of each phase all pages are unmarked.

For each request at time t for page p:

1. If p is in cache:
   - Mark p.
   - Return HIT.

2. If p is not in cache (MISS):
   a. If there are unmarked pages in cache:
      - Evict the **least-recently-used** (LRU) among the unmarked pages.
   b. If all pages are marked (no unmarked pages):
      - **Start a new phase**: unmark all cached pages.
      - Evict the LRU page among all (now unmarked) cached pages.
   - Add p to cache, mark p.
   - Return MISS.

**Phase structure**: A phase ends (and a new one begins) whenever a miss
occurs with all cached pages marked.  Equivalently, each phase sees at most
k distinct page requests before a miss triggers a phase boundary.

Paper-to-code mapping
----------------------
| Concept             | Code location                            |
|---------------------|------------------------------------------|
| Phase boundary      | ``len(unmarked) == 0`` branch in on_request |
| Marked pages        | ``_marked: Set[PageId]``                 |
| LRU tracking        | ``_lru_order: OrderedDict``              |
| Phase counter       | ``_phase_count``                         |
"""

from __future__ import annotations

import collections
from typing import Dict, Optional, Set

from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


class MarkerPolicy(BasePolicy):
    """Deterministic LRU-Marker caching policy.

    O(H_k)-competitive (worst-case) for unweighted paging.
    Ignores predictions entirely; used as the robust sub-routine in
    :class:`~lafc.policies.trust_and_doubt.TrustAndDoubtPolicy`.
    """

    name: str = "marker"

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        # Pages marked in the current phase (requested at least once this phase).
        self._marked: Set[PageId] = set()
        # LRU order for all *cached* pages (oldest → newest).
        self._lru_order: collections.OrderedDict[PageId, None] = collections.OrderedDict()
        # Diagnostic counter: number of phases completed.
        self._phase_count: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _choose_eviction_candidate(self) -> PageId:
        """Return the LRU page among unmarked cached pages.

        If no unmarked page exists, start a new phase (unmark all) and then
        return the LRU page among all (now unmarked) cached pages.
        """
        unmarked = [q for q in self._lru_order if q not in self._marked]

        if not unmarked:
            # All pages marked → phase boundary: unmark all, start new phase.
            self._marked = set()
            self._phase_count += 1
            # After unmarking, all cached pages are candidates.
            unmarked = list(self._lru_order.keys())

        if not unmarked:
            raise RuntimeError("No eviction candidate found; cache may be empty.")

        # Evict the LRU among the unmarked candidates.
        # _lru_order is maintained oldest → newest, so iterate front-to-back.
        for q in self._lru_order:  # oldest first
            if q in unmarked:
                return q
        raise RuntimeError("No LRU candidate found.")  # pragma: no cover

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id
        evicted: Optional[PageId] = None

        if self.in_cache(pid):
            # Hit: mark and move to most-recently-used end.
            self._marked.add(pid)
            self._lru_order.move_to_end(pid)
            self._record_hit()
            return CacheEvent(t=request.t, page_id=pid, hit=True, cost=0.0)

        # Cache miss.
        cost = self._pages[pid].weight
        self._record_miss(cost)

        if self._cache.is_full():
            evicted = self._choose_eviction_candidate()
            self._lru_order.pop(evicted)
            self._evict(evicted)

        # Fetch, mark, and register as most-recently-used.
        self._add(pid)
        self._marked.add(pid)
        self._lru_order[pid] = None

        return CacheEvent(t=request.t, page_id=pid, hit=False, cost=cost, evicted=evicted)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def phase_count(self) -> int:
        """Number of phases completed so far (starts at 0)."""
        return self._phase_count

    def marked_snapshot(self) -> frozenset:
        """Return a snapshot of the currently marked page ids."""
        return frozenset(self._marked)
