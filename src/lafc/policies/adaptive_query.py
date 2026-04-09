"""AdaptiveQuery-b (parsimonious learning-augmented caching).

Reference
---------
Im, Kumar, Petety, Purohit.
"Parsimonious Learning-Augmented Caching." ICML 2022, PMLR 162.

This policy implements the paper's **AdaptiveQuery-b** marking algorithm
(Algorithm 3) together with the worst-case robust chain-length cap from
Section 5.3 (Theorem 11): once an eviction chain exceeds ``log k``, the
algorithm switches to randomized-marking style random eviction.

Paper-faithful core
-------------------
- Unweighted paging with unit miss cost.
- Marker phase structure.
- On miss with full cache, choose victim from unmarked pages.
- Query up to ``b`` unmarked pages sampled uniformly without replacement,
  and evict the sampled page with largest predicted next-arrival.
- Robust fallback: when chain-depth ``j > log k``, skip querying and evict
  a uniformly random unmarked page.

Important interface adaptation (explicit)
-----------------------------------------
The paper assumes query access ``Q(p, t)`` for *any* cached page at time ``t``.
This repository's standard trace format only provides ``predicted_next`` for the
currently requested page. To remain compatible with existing interfaces, this
implementation interprets querying ``Q(p, t)`` as reading the most recently
observed predictor value for page ``p`` (updated whenever ``p`` is requested).
If no predictor value has yet been observed for ``p``, default is ``+inf``.

This is a defensible operational adapter used consistently across the current
baselines, but it is not equivalent to an oracle that can always return a
fresh page-time-specific query answer.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


@dataclass
class AdaptiveQueryDiagnostics:
    queries_used: int = 0
    misses_with_query: int = 0
    misses_with_fallback_random: int = 0
    query_mode_evictions: int = 0
    random_mode_evictions: int = 0
    sampled_pages_total: int = 0
    max_chain_depth_seen: int = 0


class AdaptiveQueryPolicy(BasePolicy):
    """AdaptiveQuery-b with robust chain-length fallback (Im et al., 2022)."""

    name: str = "adaptive_query"

    def __init__(self, b: int = 2, seed: int = 0):
        if b <= 0:
            raise ValueError(f"b must be >= 1, got {b}")
        self._b = int(b)
        self._seed = int(seed)

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        self._rng = random.Random(self._seed)

        self._marked: Set[PageId] = set()
        self._phase: int = 1

        # Most recently observed prediction for each page (query adapter).
        self._predicted_next: Dict[PageId, float] = {}

        # Chain-depth bookkeeping: if page q is evicted while serving page p
        # at depth d, then q receives depth d+1 when it next faults.
        self._eviction_depth: Dict[PageId, int] = {}

        self._diag = AdaptiveQueryDiagnostics()

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id
        self._predicted_next[pid] = request.predicted_next

        if self.in_cache(pid):
            self._marked.add(pid)
            self._record_hit()
            return CacheEvent(
                t=request.t,
                page_id=pid,
                hit=True,
                cost=0.0,
                phase=self._phase,
                diagnostics={"mode": "hit"},
            )

        self._record_miss(1.0)
        evicted: Optional[PageId] = None

        chain_depth = int(self._eviction_depth.pop(pid, 0))
        self._diag.max_chain_depth_seen = max(self._diag.max_chain_depth_seen, chain_depth)

        if self._cache.is_full():
            if len(self._marked) == self._cache.capacity:
                self._phase += 1
                self._marked = set()

            # Paper (Sec. 5.3): if j > log k then switch to random unmarked.
            threshold = math.log(max(2, self._cache.capacity))
            use_random_fallback = chain_depth > threshold

            if use_random_fallback:
                victim, sampled = self._random_unmarked_victim()
                mode = "fallback_random"
                self._diag.misses_with_fallback_random += 1
                self._diag.random_mode_evictions += 1
            else:
                victim, sampled = self._query_and_choose_victim(request.t)
                mode = "adaptive_query"
                self._diag.misses_with_query += 1
                self._diag.query_mode_evictions += 1
                self._diag.queries_used += len(sampled)
                self._diag.sampled_pages_total += len(sampled)

            evicted = victim
            self._evict(evicted)
            self._marked.discard(evicted)
            self._eviction_depth[evicted] = chain_depth + 1
        else:
            mode = "warmup"
            sampled = []

        self._add(pid)
        self._marked.add(pid)

        return CacheEvent(
            t=request.t,
            page_id=pid,
            hit=False,
            cost=1.0,
            evicted=evicted,
            phase=self._phase,
            diagnostics={
                "mode": mode,
                "chain_depth": chain_depth,
                "sampled": sampled,
            },
        )

    def _query_and_choose_victim(self, t: int) -> tuple[PageId, List[PageId]]:
        del t  # Q(p,t) adapter currently uses latest observed predictor value.
        unmarked = sorted(self._cache.current_cache() - self._marked)
        if not unmarked:
            raise RuntimeError("No unmarked page available for adaptive query eviction")

        sample_size = min(self._b, len(unmarked))
        sampled = self._rng.sample(unmarked, k=sample_size)

        # Evict argmax predicted_next among sampled pages.
        # Deterministic tie-break for reproducibility: page_id descending.
        victim = min(
            sampled,
            key=lambda q: (-self._predicted_next.get(q, math.inf), q),
        )
        return victim, sampled

    def _random_unmarked_victim(self) -> tuple[PageId, List[PageId]]:
        unmarked = sorted(self._cache.current_cache() - self._marked)
        if not unmarked:
            raise RuntimeError("No unmarked page available for random fallback eviction")
        victim = self._rng.choice(unmarked)
        return victim, []

    def diagnostics_summary(self) -> Dict[str, float]:
        miss_count = max(1, self._misses)
        return {
            "b": float(self._b),
            "seed": float(self._seed),
            "queries_used": float(self._diag.queries_used),
            "fraction_misses_queried": float(self._diag.misses_with_query) / float(miss_count),
            "fraction_misses_fallback_random": float(self._diag.misses_with_fallback_random)
            / float(miss_count),
            "avg_queries_per_queried_miss": (
                float(self._diag.queries_used) / float(max(1, self._diag.misses_with_query))
            ),
            "query_mode_evictions": float(self._diag.query_mode_evictions),
            "random_mode_evictions": float(self._diag.random_mode_evictions),
            "max_chain_depth_seen": float(self._diag.max_chain_depth_seen),
        }
