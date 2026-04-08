"""
Standard Marker caching algorithm (unweighted paging).

Reference
---------
Fiat, Karp, Luby, McGeoch, Sleator, Young.
"Competitive Paging Algorithms."  Journal of Algorithms, 1991.

Used as the deterministic backbone of Predictive Marker in:
    Lykouris, Vassilvitskii.
    "Competitive Caching with Machine Learned Advice."
    ICML 2018 / JACM 2021.

============================================================
PAPER-TO-CODE IMPLEMENTATION NOTE
============================================================

Setting
-------
Standard paging (unweighted):
- Cache capacity k (unit-size pages).
- All misses have unit cost (cost = 1).
- Request sequence σ_1, ..., σ_T.
- No predictions used by this algorithm.

Phase structure
---------------
A phase is a maximal time interval during which at most k distinct
*new* pages are faulted on.  Equivalently:

- At the start of each phase all k pages in cache are "unmarked"
  (or the cache has fewer than k pages at the very beginning).
- When a page is requested:
    - If the page is in cache: mark it (if not already marked) → HIT.
    - If the page is not in cache: MISS.
        - If all cached pages are already marked (|M| = k):
            start a new phase: unmark all pages (M ← ∅).
        - Evict an arbitrary unmarked page (we choose deterministically
          — see INTERPRETATION NOTE below).
        - Fetch the requested page and mark it.

The algorithm finishes a phase exactly when it faults on the (k+1)-th
distinct page since the last phase start.

INTERPRETATION NOTE — Eviction tie-breaking
--------------------------------------------
The original Marker paper evicts a *randomly* chosen unmarked page for
a randomised O(log k)-competitive algorithm.  This implementation is
deterministic for reproducibility: when multiple unmarked pages are
available, we evict the one with the lexicographically smallest
page_id.  This does not affect the algorithm's correctness for tracing
phase behaviour, and the competitive ratio of the *deterministic*
version is O(k) (any deterministic algorithm for paging is Ω(k) in the
worst case).

The random variant is more interesting theoretically, but deterministic
Marker is sufficient as a backbone for Predictive Marker and for
verifying phase logic in unit tests.

Maintained state
----------------
  _marked : set[PageId]
      Set of marked pages in the current phase (subset of _cache).
  _phase  : int
      Current phase index (1-based; incremented on every phase start).
"""

from __future__ import annotations

from typing import Dict, Optional, Set

from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


class MarkerPolicy(BasePolicy):
    """Deterministic Marker caching policy (unweighted).

    Phase-based algorithm: pages are marked when accessed and
    unmarked at the start of each new phase.  On a miss the policy
    evicts the lexicographically smallest unmarked page (deterministic
    tie-breaking).

    This policy ignores predictions; all costs are unit (1.0 per miss).
    """

    name: str = "marker"

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        self._marked: Set[PageId] = set()
        self._phase: int = 1

    # ------------------------------------------------------------------
    # Main algorithm step
    # ------------------------------------------------------------------

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id
        evicted: Optional[PageId] = None

        if self.in_cache(pid):
            # Cache hit: mark the page (idempotent).
            self._marked.add(pid)
            self._record_hit()
            return CacheEvent(
                t=request.t, page_id=pid, hit=True, cost=0.0, phase=self._phase
            )

        # Cache miss — pay unit cost.
        self._record_miss(1.0)

        if self._cache.is_full():
            if len(self._marked) == self._cache.capacity:
                # All k pages are marked → start a new phase.
                self._phase += 1
                self._marked = set()
            # Evict an unmarked page (deterministic: lexicographically smallest).
            evicted = self._choose_victim()
            self._marked.discard(evicted)
            self._evict(evicted)

        # Fetch and mark the requested page.
        self._add(pid)
        self._marked.add(pid)

        return CacheEvent(
            t=request.t,
            page_id=pid,
            hit=False,
            cost=1.0,
            evicted=evicted,
            phase=self._phase,
        )

    # ------------------------------------------------------------------
    # Eviction logic
    # ------------------------------------------------------------------

    def _choose_victim(self) -> PageId:
        """Return the unmarked cached page to evict.

        Deterministic tie-breaking: lexicographically smallest page_id
        among all unmarked cached pages.

        INTERPRETATION NOTE: The original Marker algorithm evicts a
        uniformly random unmarked page to achieve O(log k) competitiveness.
        We use a deterministic rule for reproducibility; the choice of
        which unmarked page to evict does not affect the theoretical
        guarantees of the *Predictive Marker* algorithm built on top.
        """
        unmarked = self._cache.current_cache() - self._marked
        if not unmarked:
            raise RuntimeError(
                "No unmarked page available to evict — this indicates a bug. "
                f"cache={self._cache.current_cache()}, marked={self._marked}"
            )
        return min(unmarked)  # lexicographically smallest

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def current_phase(self) -> int:
        """Return the current phase index (1-based)."""
        return self._phase

    def marked_pages(self) -> frozenset:
        """Return a snapshot of the currently marked page ids."""
        return frozenset(self._marked)
