"""
Deterministic BlindOracle + LRU black-box combiner (Baseline 4).

Reference
---------
Alexander Wei.
"Better and Simpler Learning-Augmented Online Caching."
APPROX/RANDOM 2020 (LIPIcs Vol. 176, Art. 60).

============================================================
STEP-0 PAPER-TO-CODE NOTE (faithfulness-oriented)
============================================================
1) Required combiner state
--------------------------
To combine BlindOracle and LRU online, we maintain:
- two shadow policies (BlindOracle and LRU),
- cumulative miss counts of both shadows,
- deterministic tie-breaking rule when shadow costs are equal,
- combiner-visible cache state (for diagnostics/output consistency).

2) BlindOracle definition
-------------------------
Unweighted paging, unit miss cost, eviction on a miss with full cache:
    evict argmax predicted_next among cached pages
with deterministic tie-breaking.

3) LRU definition
-----------------
Standard deterministic LRU paging with unit miss cost.

4) How cumulative costs are tracked
-----------------------------------
Each shadow processes every request.  Their private miss counters are read
before each request and used for the combiner decision at that request.

5) Operational meaning of
   "follow whichever has performed better so far"
--------------------------------------------------
INTERPRETATION NOTE:
The paper gives this rule informally and cites black-box combiner machinery.
To preserve the literal "follow the algorithm" semantics (and avoid a custom
third-cache hybrid rule), this implementation does **not** apply BO/LRU rules
on an independent combiner cache.  Instead, at request t it follows the shadow
algorithm whose cumulative misses up to t-1 are lower (tie -> BlindOracle).
The combiner event at t is taken from that selected shadow's action.

6) Tie-breaking
---------------
If both shadows have equal cumulative misses, choose BlindOracle.
This deterministic choice is documented and reproducible.

7) Was previous implementation faithful?
----------------------------------------
Previous code kept an independent combiner cache and applied whichever
*eviction rule* was leading to that third cache.  That is a plausible
interpretation, but it is not a literal "follow the leading algorithm"
implementation.  This file now implements the latter explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from lafc.policies.base import BasePolicy
from lafc.policies.blind_oracle import BlindOraclePolicy
from lafc.policies.lru import LRUPolicy
from lafc.simulator.cache_state import CacheState
from lafc.types import CacheEvent, Page, PageId, Request


@dataclass
class CombinerStepLog:
    """Per-step diagnostics for the deterministic combiner."""

    t: int
    page_id: PageId
    hit: bool
    evicted: Optional[PageId]
    chosen: str
    bo_misses_before: int
    lru_misses_before: int


class BlindOracleLRUCombiner(BasePolicy):
    """Deterministic BlindOracle + LRU combiner for Baseline 4.

    The policy keeps two shadow simulations and follows the one with lower
    cumulative miss count so far (tie -> BlindOracle).
    """

    name: str = "blind_oracle_lru_combiner"

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)

        self._shadow_bo = BlindOraclePolicy()
        self._shadow_bo.reset(capacity, pages)

        self._shadow_lru = LRUPolicy()
        self._shadow_lru.reset(capacity, pages)

        self._step_log: List[CombinerStepLog] = []

        # The currently selected algorithm for serving requests.
        self._active: str = "blind_oracle"

    def _select_active(self, bo_misses: int, lru_misses: int) -> str:
        # Tie broken in favor of BlindOracle.
        return "blind_oracle" if bo_misses <= lru_misses else "lru"

    def _shadow_event(self, which: str, request: Request) -> CacheEvent:
        if which == "blind_oracle":
            return self._shadow_bo.on_request(request)
        if which == "lru":
            return self._shadow_lru.on_request(request)
        raise ValueError(f"Unknown shadow '{which}'")

    def _sync_visible_cache(self, which: str) -> None:
        """Mirror visible cache state to chosen shadow for diagnostics.

        INTERPRETATION NOTE:
        This model treats the combiner as literally following one of two
        black-box shadows; the visible cache is mirrored to the selected
        shadow cache after every request.
        """
        target = (
            self._shadow_bo.current_cache()
            if which == "blind_oracle"
            else self._shadow_lru.current_cache()
        )
        new_cache = CacheState(self._cache.capacity, self._pages)
        for pid in sorted(target):
            new_cache.add(pid)
        self._cache = new_cache

    def on_request(self, request: Request) -> CacheEvent:
        bo_before = self._shadow_bo._misses
        lru_before = self._shadow_lru._misses

        chosen = self._select_active(bo_before, lru_before)
        self._active = chosen

        chosen_event = self._shadow_event(chosen, request)
        # Keep the non-chosen shadow updated on every request as well.
        other = "lru" if chosen == "blind_oracle" else "blind_oracle"
        self._shadow_event(other, request)

        if chosen_event.hit:
            self._record_hit()
        else:
            self._record_miss(1.0)

        self._step_log.append(
            CombinerStepLog(
                t=request.t,
                page_id=request.page_id,
                hit=chosen_event.hit,
                evicted=chosen_event.evicted,
                chosen=chosen,
                bo_misses_before=bo_before,
                lru_misses_before=lru_before,
            )
        )

        # Mirror visible cache to whichever algorithm is better *after* this
        # request, so next request starts from the current leader.
        next_active = self._select_active(self._shadow_bo._misses, self._shadow_lru._misses)
        self._sync_visible_cache(next_active)
        self._active = next_active

        return CacheEvent(
            t=request.t,
            page_id=request.page_id,
            hit=chosen_event.hit,
            cost=0.0 if chosen_event.hit else 1.0,
            evicted=chosen_event.evicted,
            diagnostics={
                "active_before": chosen,
                "active_after": next_active,
            },
        )

    def step_log(self) -> List[CombinerStepLog]:
        return list(self._step_log)

    def shadow_bo_misses(self) -> int:
        return self._shadow_bo._misses

    def shadow_lru_misses(self) -> int:
        return self._shadow_lru._misses
