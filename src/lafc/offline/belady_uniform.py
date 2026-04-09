"""Exact offline Belady solver for the uniform paging setting."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Set

from lafc.offline.types import OfflineDecision, OfflineSimulationResult
from lafc.offline.validation import validate_uniform_paging_inputs
from lafc.types import Page, PageId, Request


@dataclass
class BeladyUniformPagingSolver:
    """Belady (furthest-in-future) for equal-size, equal-cost paging."""

    name: str = "offline_belady_uniform"

    def solve(
        self,
        requests: Iterable[Request],
        pages: Dict[PageId, Page],
        capacity: int,
        *,
        validation_mode: str = "strict",
    ) -> OfflineSimulationResult:
        req_list = list(requests)
        if capacity <= 0:
            raise ValueError(f"Cache capacity must be >= 1, got {capacity}")

        validation = validate_uniform_paging_inputs(req_list, pages, mode=validation_mode)

        future_positions: Dict[PageId, Deque[int]] = defaultdict(deque)
        for idx, req in enumerate(req_list):
            future_positions[req.page_id].append(idx)

        cache: Set[PageId] = set()
        decisions: List[OfflineDecision] = []
        total_hits = 0
        total_misses = 0
        tie_count = 0
        evicted_never_used_again_count = 0

        for t, req in enumerate(req_list):
            pid = req.page_id
            future_positions[pid].popleft()

            if pid in cache:
                total_hits += 1
                decisions.append(OfflineDecision(t=t, page_id=pid, hit=True, cost=0.0))
                continue

            total_misses += 1

            evicted = None
            evicted_next_use = None
            evicted_distance = None
            evicted_never_used_again = False
            tie_size = 1

            if len(cache) >= capacity:
                candidate_next: Dict[PageId, int] = {
                    q: (future_positions[q][0] if future_positions[q] else math.inf)
                    for q in cache
                }
                farthest = max(candidate_next.values())
                tied = sorted([q for q, nxt in candidate_next.items() if nxt == farthest])
                tie_size = len(tied)
                if tie_size > 1:
                    tie_count += 1

                victim = tied[0]
                evicted = victim
                next_use_raw = candidate_next[victim]
                evicted_next_use = None if math.isinf(next_use_raw) else int(next_use_raw)
                evicted_distance = (
                    None if math.isinf(next_use_raw) else int(next_use_raw) - t
                )
                evicted_never_used_again = math.isinf(next_use_raw)
                if evicted_never_used_again:
                    evicted_never_used_again_count += 1

                cache.remove(victim)

            cache.add(pid)
            decisions.append(
                OfflineDecision(
                    t=t,
                    page_id=pid,
                    hit=False,
                    cost=1.0,
                    evicted=evicted,
                    evicted_next_use=evicted_next_use,
                    evicted_next_use_distance=evicted_distance,
                    evicted_never_used_again=evicted_never_used_again,
                    tie_size=tie_size,
                )
            )

        diagnostics = {
            "validation": {
                "mode": validation.mode,
                "is_uniform": validation.is_uniform,
                "unique_weights": validation.unique_weights,
                "representative_weight": validation.representative_weight,
            },
            "ties_on_eviction": tie_count,
            "evictions_never_used_again": evicted_never_used_again_count,
            "total_evictions": sum(1 for d in decisions if d.evicted is not None),
        }

        return OfflineSimulationResult(
            solver_name=self.name,
            capacity=capacity,
            total_requests=len(req_list),
            total_hits=total_hits,
            total_misses=total_misses,
            decisions=decisions,
            diagnostics=diagnostics,
        )
