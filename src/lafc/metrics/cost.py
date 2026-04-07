"""
Cache cost accounting utilities.
"""

from __future__ import annotations

from typing import Dict, List

from lafc.types import CacheEvent, PageId


def total_fetch_cost(events: List[CacheEvent]) -> float:
    """Sum of fetch costs across all cache misses."""
    return sum(e.cost for e in events)


def total_hits(events: List[CacheEvent]) -> int:
    """Number of cache hits."""
    return sum(1 for e in events if e.hit)


def total_misses(events: List[CacheEvent]) -> int:
    """Number of cache misses (faults)."""
    return sum(1 for e in events if not e.hit)


def hit_rate(events: List[CacheEvent]) -> float:
    """Fraction of requests that were cache hits.  Returns 0.0 for empty traces."""
    n = len(events)
    if n == 0:
        return 0.0
    return total_hits(events) / n


def per_page_cost(events: List[CacheEvent]) -> Dict[PageId, float]:
    """Return the total fetch cost incurred for each page id."""
    result: Dict[PageId, float] = {}
    for e in events:
        if not e.hit:
            result[e.page_id] = result.get(e.page_id, 0.0) + e.cost
    return result
