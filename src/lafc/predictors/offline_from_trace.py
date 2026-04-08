"""Offline-derived predictors for paging traces."""

from __future__ import annotations

import math
from typing import Dict, List, Set

from lafc.types import PageId, Request


def extract_actual_next_arrivals(page_ids: List[PageId]) -> List[float]:
    n = len(page_ids)
    result: List[float] = [math.inf] * n
    last_seen: Dict[PageId, int] = {}
    for t in range(n - 1, -1, -1):
        pid = page_ids[t]
        if pid in last_seen:
            result[t] = float(last_seen[pid])
        last_seen[pid] = t
    return result


def compute_perfect_predictions(requests: List[Request]) -> List[Request]:
    return [
        Request(
            t=req.t,
            page_id=req.page_id,
            predicted_next=req.actual_next,
            actual_next=req.actual_next,
            metadata=dict(req.metadata),
        )
        for req in requests
    ]


def build_predicted_caches_from_next_arrival(requests: List[Request], capacity: int) -> List[Set[PageId]]:
    """Convert L&V-style next-arrival advice to MTS-style cache predictions P_t.

    Implements the conversion described in Sec. 1.3 of Antoniadis et al. (ICML 2020):
    run Blind Oracle on predicted-next advice and use its cache at each time as P_t.
    """
    cache: Set[PageId] = set()
    pred_next_by_page: Dict[PageId, float] = {}
    out: List[Set[PageId]] = []

    for req in requests:
        pred_next_by_page[req.page_id] = req.predicted_next
        if req.page_id not in cache:
            if len(cache) == capacity:
                victim = max(cache, key=lambda q: (pred_next_by_page.get(q, math.inf), q))
                cache.remove(victim)
            cache.add(req.page_id)
        out.append(set(cache))
    return out


def attach_predicted_caches(requests: List[Request], capacity: int) -> List[Request]:
    caches = build_predicted_caches_from_next_arrival(requests, capacity)
    enriched: List[Request] = []
    for req, cfg in zip(requests, caches):
        md = dict(req.metadata)
        md["predicted_cache"] = sorted(cfg)
        enriched.append(
            Request(
                t=req.t,
                page_id=req.page_id,
                predicted_next=req.predicted_next,
                actual_next=req.actual_next,
                metadata=md,
            )
        )
    return enriched
