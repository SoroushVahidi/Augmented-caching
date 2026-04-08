"""TRUST&DOUBT for unweighted paging (Antoniadis et al., ICML 2020, Alg. 3).

PAPER-TO-CODE NOTE (Step 0)
---------------------------
1) Setting: unweighted paging with cache size k.
2) Predictions: predictor cache configuration P_t (lazy predictor); in code
   this is ``request.metadata['predicted_cache']``.
3) State: A, stale, C, T, D, U, M; per-clean-page p_q, trusted(q), t_q,
   q_interval_change; plus random priority order on U when A empties.
4) Logic: implemented from Algorithm 3 (supplementary, page with Alg. 3).
5) Difference from Blind Oracle / Predictive Marker:
   - Blind Oracle always trusts one-step eviction advice.
   - Predictive Marker stays marking-only and evicts unmarked only.
   - TRUST&DOUBT can evict marked pages in T and adaptively doubles thresholds.
6) Ambiguities interpreted:
   - "arbitrary" choices use seeded random choices for reproducibility.
   - non-lazy algorithm is simulated in background; real cache is lazy per Remark 10.
7) Mapping: on_request implements Alg.3 blocks (phase start, step1..4).
8) Previous implementation status:
   - Partly interpreted and too deterministic in "arbitrary" branches.
   - This version restores randomized arbitrary choices while preserving
     deterministic replay under a fixed seed.

INTERPRETATION NOTE: where the paper leaves tie-breaking unspecified, seeded
random selection is used and documented for reproducibility.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Optional, Set

from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


class TrustAndDoubtPolicy(BasePolicy):
    name = "trust_and_doubt"

    def __init__(self, seed: int = 0):
        self._seed = seed

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        self._rng = random.Random(self._seed)
        self._phase = 1
        self._last_touch: Dict[PageId, int] = defaultdict(lambda: -10**9)

        # lazy real cache from BasePolicy's _cache.
        # simulated non-lazy cache in the paper (Remark 10):
        self._sim_cache: Set[PageId] = set()

        self._marked: Set[PageId] = set()
        self._A: Set[PageId] = set()
        self._stale: Set[PageId] = set()
        self._U: Set[PageId] = set()
        self._M: Set[PageId] = set()
        self._C: Set[PageId] = set()
        self._T: Set[PageId] = set()
        self._D: Set[PageId] = set()

        self._pq: Dict[PageId, PageId] = {}
        self._trusted: Dict[PageId, bool] = {}
        self._tq: Dict[PageId, int] = {}
        self._q_interval_change: Dict[PageId, int] = {}

        self._priority: Dict[PageId, int] = {}

    def _predicted_cache(self, request: Request) -> Set[PageId]:
        pc = request.metadata.get("predicted_cache")
        if pc is None:
            raise ValueError(
                "TRUST&DOUBT requires metadata['predicted_cache']; "
                "use --derive-predicted-caches or provide predicted_caches in trace"
            )
        P = set(str(x) for x in pc)
        if len(P) > self._cache.capacity:
            raise ValueError(
                f"predicted_cache size {len(P)} exceeds capacity {self._cache.capacity}"
            )
        return P

    def _lru_pick(self, pages: Set[PageId]) -> PageId:
        return min(pages, key=lambda p: (self._last_touch[p], p))

    def _random_pick(self, pages: Set[PageId]) -> PageId:
        """Pick an arbitrary page using seeded randomness."""
        items = sorted(pages)
        return items[self._rng.randrange(len(items))]

    def _start_new_phase_if_needed(self, r: PageId) -> None:
        if len(self._marked) == self._cache.capacity and r not in self._marked:
            self._phase += 1
            self._A = {p for p in self._sim_cache if p not in self._marked}
            self._stale = set(self._sim_cache) - self._A
            self._marked = set()
            self._U = set(self._stale)
            self._M = set()
            self._T = set()
            self._D = set()
            self._C = set()
            self._pq = {}
            self._trusted = {}
            self._tq = {}
            self._q_interval_change = {}
            self._priority = {}

    def _assign_random_priorities(self) -> None:
        items = list(self._U)
        self._rng.shuffle(items)
        self._priority = {p: i for i, p in enumerate(items)}

    def _pick_missing_from_predictor(self, predictor_cache: Set[PageId]) -> PageId:
        candidates = ((self._U | self._M) - self._D) - predictor_cache
        if not candidates:
            # INTERPRETATION NOTE: if no page is currently missing from predictor
            # cache (can happen with imperfect conversion/warm-up), fall back to
            # arbitrary page from (U ∪ M) \ D as suggested by "arbitrary" clauses.
            candidates = (self._U | self._M) - self._D
            if not candidates:
                raise RuntimeError("No candidate page available for p_q definition")
        # INTERPRETATION NOTE: paper says "arbitrary"; we use seeded random.
        return self._random_pick(candidates)

    def _sync_real_cache_on_fault(self, requested: PageId) -> Optional[PageId]:
        """Lazy implementation per Remark 10.

        On a real miss, evict any page in real cache but not in simulated cache.
        """
        evicted = None
        if self.in_cache(requested):
            return None
        if self._cache.is_full():
            diff = set(self._cache.current_cache()) - self._sim_cache
            victim = self._lru_pick(diff if diff else set(self._cache.current_cache()))
            evicted = victim
            self._evict(victim)
        self._add(requested)
        return evicted

    def on_request(self, request: Request) -> CacheEvent:
        r = request.page_id
        P = self._predicted_cache(request)
        self._last_touch[r] = request.t

        if not self._sim_cache:
            self._sim_cache = set(self._cache.current_cache())

        self._start_new_phase_if_needed(r)

        is_arrival = r not in self._marked
        if is_arrival:
            self._marked.add(r)
            if r not in self._T:
                self._M.add(r)
            self._U.discard(r)

        if self._A:
            if r in self._A:
                self._A.remove(r)
            elif r not in self._sim_cache:
                # INTERPRETATION NOTE: paper says arbitrary ancient page.
                anc = self._random_pick(self._A)
                self._A.remove(anc)
                self._sim_cache.remove(anc)
                self._sim_cache.add(r)
            if not self._A and not self._priority:
                self._assign_random_priorities()
        else:
            # step 1
            if r not in self._sim_cache:
                if len(self._sim_cache) < self._cache.capacity:
                    # INTERPRETATION NOTE: during initial warm-up the supplement
                    # pseudocode assumes full cache; we directly load r.
                    self._sim_cache.add(r)
                else:
                    in_u = self._U & self._sim_cache
                    if not in_u:
                        raise RuntimeError("step 1 expected a victim in U ∩ cache")
                    # Lowest priority (i.e. largest rank) is evicted.
                    victim = max(in_u, key=lambda p: (self._priority.get(p, 10**9), p))
                    self._sim_cache.remove(victim)
                    self._sim_cache.add(r)

            # step 2
            if len(self._sim_cache) == self._cache.capacity and r not in self._stale and is_arrival:
                self._C.add(r)
                self._pq[r] = self._pick_missing_from_predictor(P)
                self._trusted[r] = True
                self._T.add(self._pq[r])
                self._U.discard(self._pq[r])
                self._M.discard(self._pq[r])
                self._tq[r] = 1
                self._q_interval_change[r] = len(self._marked)

            # step 3
            for q in list(self._C):
                if self._pq.get(q) == r:
                    old = self._pq[q]
                    self._D.discard(old)
                    self._T.discard(old)
                    if old in self._marked:
                        self._M.add(old)
                    else:
                        self._U.add(old)
                    self._pq[q] = self._pick_missing_from_predictor(P)
                    self._trusted[q] = False
                    self._D.add(self._pq[q])

            # step 4
            for q in list(self._C):
                if is_arrival and self._q_interval_change[q] == len(self._marked):
                    if not self._trusted[q]:
                        self._tq[q] = 2 * self._tq[q]
                    self._trusted[q] = True
                    self._T.add(self._pq[q])
                    self._D.discard(self._pq[q])
                    self._U.discard(self._pq[q])
                    self._M.discard(self._pq[q])
                    self._q_interval_change[q] = len(self._marked) + self._tq[q]

                    if self._pq[q] in self._sim_cache:
                        missing_u = self._U - self._sim_cache
                        if missing_u:
                            p = min(missing_u, key=lambda x: (self._priority.get(x, 10**9), x))
                            self._sim_cache.remove(self._pq[q])
                            self._sim_cache.add(p)

        # serve request with lazy real cache
        real_hit = self.in_cache(r)
        if real_hit:
            self._record_hit()
            cost = 0.0
            evicted = None
        else:
            self._record_miss(1.0)
            cost = 1.0
            evicted = self._sync_real_cache_on_fault(r)

        return CacheEvent(
            t=request.t,
            page_id=r,
            hit=real_hit,
            cost=cost,
            evicted=evicted,
            phase=self._phase,
            diagnostics={
                "sim_cache": sorted(self._sim_cache),
                "A": sorted(self._A),
                "U": sorted(self._U),
                "M": sorted(self._M),
                "T": sorted(self._T),
                "D": sorted(self._D),
                "C": sorted(self._C),
                "pq": {k: self._pq[k] for k in sorted(self._pq)},
                "trusted": {k: self._trusted[k] for k in sorted(self._trusted)},
                "tq": {k: self._tq[k] for k in sorted(self._tq)},
                "q_interval_change": {
                    k: self._q_interval_change[k]
                    for k in sorted(self._q_interval_change)
                },
            },
        )
