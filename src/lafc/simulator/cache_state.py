"""
Cache state manager.

Keeps track of which pages are currently in cache and enforces the
capacity constraint.  This is a pure bookkeeping class; eviction
decisions are made by the policy layer.
"""

from __future__ import annotations

from typing import Dict, FrozenSet

from lafc.types import Page, PageId


class CacheState:
    """Manages the set of cached pages.

    Parameters
    ----------
    capacity:
        Maximum number of pages that may reside in cache simultaneously.
    pages:
        Mapping of all known pages (used only for validation).
    """

    def __init__(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        if capacity <= 0:
            raise ValueError(f"Cache capacity must be >= 1, got {capacity}")
        self._capacity: int = capacity
        self._pages: Dict[PageId, Page] = pages
        self._cache: Dict[PageId, None] = {}  # ordered-insertion dict used as ordered set

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def in_cache(self, page_id: PageId) -> bool:
        """Return True if *page_id* is currently cached."""
        return page_id in self._cache

    def current_cache(self) -> FrozenSet[PageId]:
        """Return an immutable snapshot of the currently cached page ids."""
        return frozenset(self._cache)

    def is_full(self) -> bool:
        """Return True when the cache has reached its capacity."""
        return len(self._cache) >= self._capacity

    def size(self) -> int:
        """Return the current number of pages in cache."""
        return len(self._cache)

    @property
    def capacity(self) -> int:
        return self._capacity

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add(self, page_id: PageId) -> None:
        """Add *page_id* to the cache.

        Raises
        ------
        KeyError
            If *page_id* is not in the pages dictionary.
        ValueError
            If the cache is already full and *page_id* is not already cached.
        """
        if page_id not in self._pages:
            raise KeyError(f"Unknown page '{page_id}'")
        if page_id in self._cache:
            return  # already present – no-op
        if self.is_full():
            raise ValueError(
                f"Cannot add page '{page_id}': cache is full "
                f"({len(self._cache)}/{self._capacity})"
            )
        self._cache[page_id] = None

    def evict(self, page_id: PageId) -> None:
        """Remove *page_id* from the cache.

        Raises
        ------
        KeyError
            If *page_id* is not currently in cache.
        """
        if page_id not in self._cache:
            raise KeyError(f"Cannot evict page '{page_id}': not in cache")
        del self._cache[page_id]
