"""
Abstract base class for all caching policies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, FrozenSet

from lafc.simulator.cache_state import CacheState
from lafc.types import CacheEvent, Page, PageId, Request


class BasePolicy(ABC):
    """Interface that every caching policy must satisfy.

    Subclasses implement :meth:`on_request`; the base class provides thin
    helpers that delegate to the underlying :class:`~lafc.simulator.cache_state.CacheState`.
    """

    name: str = "base"

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        """Initialise (or re-initialise) the policy state.

        Must be called before the first :meth:`on_request`.

        Parameters
        ----------
        capacity:
            Cache size in number of pages.
        pages:
            Dictionary of all pages that may appear in the trace.
        """
        self._pages: Dict[PageId, Page] = pages
        self._cache: CacheState = CacheState(capacity, pages)
        self._cost: float = 0.0
        self._hits: int = 0
        self._misses: int = 0

    @abstractmethod
    def on_request(self, request: Request) -> CacheEvent:
        """Process a single page request and return the resulting event.

        Implementations MUST:
        - check whether the page is already in cache (hit/miss),
        - on a miss, pay the fetch cost and update ``_cost``,
        - on a miss when cache is full, call :meth:`_evict` before adding,
        - call :meth:`_add` to insert the fetched page,
        - return a :class:`~lafc.types.CacheEvent`.
        """

    # ------------------------------------------------------------------
    # Convenience helpers (used by subclasses)
    # ------------------------------------------------------------------

    def in_cache(self, page_id: PageId) -> bool:
        return self._cache.in_cache(page_id)

    def current_cache(self) -> FrozenSet[PageId]:
        return self._cache.current_cache()

    def total_cost(self) -> float:
        return self._cost

    def _add(self, page_id: PageId) -> None:
        self._cache.add(page_id)

    def _evict(self, page_id: PageId) -> None:
        self._cache.evict(page_id)

    def _record_hit(self) -> None:
        self._hits += 1

    def _record_miss(self, weight: float) -> None:
        self._misses += 1
        self._cost += weight
