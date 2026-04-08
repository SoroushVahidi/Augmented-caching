"""
TRUST&DOUBT caching policy.

===========================================================================
PAPER-TO-CODE IMPLEMENTATION NOTE
===========================================================================

Paper
-----
Antoniadis, Coester, Eliáš, Polak, Simon.
"Online Metric Algorithms with Untrusted Predictions."
ICML 2020.  arXiv:2003.03033.

1. CACHING SETTING HANDLED BY TRUST&DOUBT
------------------------------------------
- Standard unweighted paging: cache of k pages, all pages have unit size
  and unit fetch cost.
- At each time step t, request σ_t arrives together with a prediction τ_t:
  the predicted next time index at which σ_t will be requested again.
  τ_t = ∞ means "predicted to never appear again."
- Objective: minimise total number of cache misses (= total fetch cost for
  unit costs).

2. ROLE OF PREDICTIONS
-----------------------
The predictor provides ``predicted_next`` for each requested page.  These
are used exclusively in the TRUST phase: the algorithm evicts the cached
page with the largest ``predicted_next`` (Blind Oracle / FTP rule).  In the
DOUBT phase, predictions are ignored and the Marker algorithm governs
evictions.  Predictions are *always* recorded (both phases), so that when
the algorithm re-enters the TRUST phase, the most recent prediction for
every page is available.

3. MAINTAINED STATE VARIABLES
-------------------------------
- ``_mode : str``
    Current operating mode: ``"trust"`` or ``"doubt"``.
- ``_predicted_next : Dict[PageId, float]``
    Most recently received ``predicted_next`` for every page.
    Initialised to ``math.inf`` for all pages.
- ``_last_access : Dict[PageId, int]``
    Most recent request time index for every page.
    Used as LRU tiebreaker in the DOUBT phase.
    Initialised to ``-1`` for all pages.
- ``_trust_budget : int``
    Maximum number of cache faults allowed in the current TRUST phase
    before switching to DOUBT.  Starts at ``initial_trust_budget`` (default k).
- ``_trust_phase_faults : int``
    Number of cache faults incurred in the current TRUST phase.
    Reset to 0 at the start of each TRUST phase.
- ``_marked : Set[PageId]``
    Pages marked (requested) since the start of the current DOUBT phase.
    Used to determine whether the Marker phase has completed.
- ``_epoch : int``
    Counts the number of completed DOUBT phases (i.e., epochs).
    Used for diagnostics.

4. EVICTION / UPDATE LOGIC
----------------------------
**TRUST phase** (follow the predictor):

    On request t for page p with prediction τ_t:
    1. Update _predicted_next[p] ← τ_t; _last_access[p] ← t.
    2. If p ∈ cache: HIT.  No mode change.
    3. If p ∉ cache: MISS.
       a. Increment _trust_phase_faults.
       b. If cache full: evict q* = argmax_{q in cache, q≠p} _predicted_next[q]
          (FTP rule; break ties by page_id).
       c. Add p to cache.
       d. If _trust_phase_faults ≥ _trust_budget:
          → Switch to DOUBT: _mode ← "doubt", _marked ← {}, reset faults.

**DOUBT phase** (Marker algorithm):

    On request t for page p with prediction τ_t:
    1. Update _predicted_next[p] ← τ_t; _last_access[p] ← t.
    2. If p ∈ cache: HIT.  Mark p: _marked ← _marked ∪ {p}.
    3. If p ∉ cache: MISS.
       a. Let unmarked = {q ∈ cache : q ∉ _marked}.
       b. If unmarked ≠ ∅:
          - Evict q* = LRU page in unmarked
            (argmin_{q in unmarked} _last_access[q]).
          - Add p, mark p.
       c. If unmarked = ∅ (all pages marked → Marker phase complete):
          → Switch to TRUST: _mode ← "trust", _epoch += 1,
            _trust_budget *= 2, _trust_phase_faults ← 0.
          → Handle this miss in trust mode (FTP eviction).
          → Increment _trust_phase_faults.
          → Check if new trust budget already exhausted (edge case).

5. HOW TRUST&DOUBT DIFFERS FROM BLIND ORACLE / PREDICTIVE MARKER
------------------------------------------------------------------
- **Blind Oracle / FTP**: always trusts predictions, no robustness mechanism.
  TRUST&DOUBT adds a budget-limited trust phase and falls back to Marker.
- **Predictive Marker**: uses predictions only to guide *which unmarked page*
  to evict within the standard Marker phase structure.  There is no explicit
  TRUST phase; the algorithm is always in "Marker mode."
  TRUST&DOUBT by contrast has a dedicated TRUST phase that uses FTP eviction
  (not Marker marking), providing O(1)-consistency when predictions are good.

6. AMBIGUITIES AND INTERPRETATIONS
------------------------------------
See INTERPRETATION NOTEs in the code.  Documented ambiguities:

  a. **Trust budget initialisation**: The paper does not specify the initial
     value of the trust budget.  We use ``k`` (the cache capacity) as the
     starting budget.  This means the first DOUBT phase is triggered after
     exactly k misses in trust mode.  INTERPRETATION NOTE A.

  b. **Trust budget update rule**: The paper's TRUST&DOUBT uses a doubling
     scheme (budget doubles after each DOUBT phase / epoch) to bound overhead
     to O(log k) multiplicative factor.  We implement this doubling.
     INTERPRETATION NOTE B.

  c. **DOUBT phase termination**: We end the DOUBT phase when a miss occurs
     with *all currently cached pages* marked.  This is equivalent to the
     standard Marker phase boundary: "all k cached pages were requested since
     the last phase start."  INTERPRETATION NOTE C.

  d. **Transition step handling**: When the DOUBT phase ends during a miss
     (all pages marked), we switch to TRUST mode and handle the current miss
     using FTP eviction — not Marker eviction.  This avoids starting a new
     Marker phase mid-DOUBT.  INTERPRETATION NOTE D.

  e. **Weighted pages**: The theoretical guarantees (consistency / robustness)
     from the ICML 2020 paper apply to unit-weight (unweighted) paging only.
     The implementation accepts arbitrary weights for generality, but
     correctness under non-unit weights is not guaranteed by the paper.
     INTERPRETATION NOTE E.

7. PAPER SECTIONS / THEOREMS
------------------------------
- Setting:        Section 2 of the paper.
- FTP (trust):    Section 3, Algorithm FTP (specialised to caching).
- Marker (doubt): Standard O(H_k)-competitive Marker (background, Section 2).
- TRUST&DOUBT:    Section 3 / Algorithm 1 (the caching specialisation).
- Consistency:    Theorem 3.3 (caching), O(1 + η / OPT).
- Robustness:     Theorem 3.3 (caching), O(log k).
- Error measure η: Definition 2.1 / Section 2 (adapted to caching).

NOTE: The exact theorem numbering may differ across arXiv versions.
      The ICML 2020 published proceedings version should be consulted.

===========================================================================
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Set

from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request

logger = logging.getLogger(__name__)


class TrustAndDoubtPolicy(BasePolicy):
    """TRUST&DOUBT caching algorithm (Antoniadis et al., ICML 2020).

    Combines a prediction-following phase (TRUST / FTP) with a robust phase
    (DOUBT / Marker) to achieve near-optimal consistency when predictions are
    accurate and O(log k)-robustness when they are adversarial.

    Parameters
    ----------
    initial_trust_budget:
        Number of cache faults allowed in the first TRUST phase before
        switching to DOUBT.  If ``None`` (default), the cache capacity ``k``
        is used.  After each DOUBT phase the budget doubles.

        INTERPRETATION NOTE A: The paper does not specify the initial budget
        value.  Using k is a natural default because one Marker phase handles
        exactly k distinct pages, so budget = k makes each TRUST phase
        comparable in length to one Marker phase.
    """

    name: str = "trust_and_doubt"

    def __init__(self, initial_trust_budget: Optional[int] = None) -> None:
        self._cfg_initial_trust_budget = initial_trust_budget

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)

        # ---- prediction state (updated at every request, both modes) ----
        # INTERPRETATION NOTE E: predictions are stored for all pages,
        # but only used for eviction decisions in TRUST mode.
        self._predicted_next: Dict[PageId, float] = {
            pid: math.inf for pid in pages
        }

        # Last access timestamp per page (for LRU tiebreaking in DOUBT mode).
        # Initialised to -1 (never accessed).
        self._last_access: Dict[PageId, int] = {pid: -1 for pid in pages}

        # ---- mode state ----
        self._mode: str = "trust"  # "trust" | "doubt"

        # INTERPRETATION NOTE A: initial budget = k (cache capacity).
        self._trust_budget: int = (
            self._cfg_initial_trust_budget
            if self._cfg_initial_trust_budget is not None
            else capacity
        )
        self._trust_phase_faults: int = 0  # faults in current TRUST phase

        # ---- DOUBT phase state ----
        # Pages marked (requested) since the start of the current DOUBT phase.
        # An empty set at the start of each DOUBT phase.
        self._marked: Set[PageId] = set()

        # Epoch counter: incremented each time a DOUBT phase completes.
        # INTERPRETATION NOTE B: used to track trust_budget doubling.
        self._epoch: int = 0

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id

        # Always update prediction and LRU timestamp (both modes).
        self._predicted_next[pid] = request.predicted_next
        self._last_access[pid] = request.t

        if self._mode == "trust":
            return self._handle_trust(request)
        else:
            return self._handle_doubt(request)

    # ------------------------------------------------------------------
    # TRUST phase
    # ------------------------------------------------------------------

    def _handle_trust(self, request: Request) -> CacheEvent:
        """Process a request in TRUST (FTP) mode.

        On a hit: no mode change.
        On a miss: FTP eviction; if budget exhausted, switch to DOUBT.
        """
        pid = request.page_id

        if self.in_cache(pid):
            self._record_hit()
            logger.debug("t=%d TRUST HIT  %s", request.t, pid)
            return CacheEvent(t=request.t, page_id=pid, hit=True, cost=0.0)

        # Cache miss in TRUST mode.
        cost = self._pages[pid].weight
        self._record_miss(cost)
        self._trust_phase_faults += 1

        evicted: Optional[PageId] = None
        if self._cache.is_full():
            evicted = self._ftp_evict(exclude=pid)
            self._evict(evicted)

        self._add(pid)

        logger.debug(
            "t=%d TRUST MISS %s | evicted=%s | faults=%d/%d | epoch=%d",
            request.t, pid, evicted, self._trust_phase_faults,
            self._trust_budget, self._epoch,
        )

        # Check if trust budget is exhausted → switch to DOUBT.
        if self._trust_phase_faults >= self._trust_budget:
            self._start_doubt_phase()

        return CacheEvent(t=request.t, page_id=pid, hit=False, cost=cost, evicted=evicted)

    # ------------------------------------------------------------------
    # DOUBT phase
    # ------------------------------------------------------------------

    def _handle_doubt(self, request: Request) -> CacheEvent:
        """Process a request in DOUBT (Marker) mode.

        On a hit: mark the page.
        On a miss with unmarked pages: Marker eviction (LRU among unmarked).
        On a miss with all pages marked: DOUBT phase complete → switch to
        TRUST and handle this miss with FTP eviction.

        INTERPRETATION NOTE C: The DOUBT phase ends when a miss occurs and
        all currently cached pages are marked, i.e., every page currently in
        cache has been requested at least once since the DOUBT phase started.
        This is the standard Marker phase boundary condition.

        INTERPRETATION NOTE D: When the transition occurs mid-miss, we switch
        to TRUST and handle the current miss in TRUST mode (FTP eviction) so
        that the current step is consistent with the new mode.
        """
        pid = request.page_id

        if self.in_cache(pid):
            self._marked.add(pid)
            self._record_hit()
            logger.debug("t=%d DOUBT HIT  %s | marked=%d", request.t, pid, len(self._marked))
            return CacheEvent(t=request.t, page_id=pid, hit=True, cost=0.0)

        # Cache miss in DOUBT mode.
        cost = self._pages[pid].weight
        self._record_miss(cost)

        evicted: Optional[PageId] = None

        if self._cache.is_full():
            unmarked = [
                q for q in self._cache.current_cache() if q not in self._marked
            ]

            if unmarked:
                # Normal Marker eviction: LRU among unmarked pages.
                evicted = self._marker_evict(unmarked)
                self._evict(evicted)
                self._add(pid)
                self._marked.add(pid)
                logger.debug(
                    "t=%d DOUBT MISS %s | evicted=%s (marker) | marked=%d",
                    request.t, pid, evicted, len(self._marked),
                )

            else:
                # INTERPRETATION NOTE C & D:
                # All cached pages are marked → Marker phase complete.
                # End DOUBT, switch to TRUST, then handle this miss with FTP.
                logger.debug(
                    "t=%d DOUBT→TRUST transition (all marked) | epoch=%d",
                    request.t, self._epoch,
                )
                self._end_doubt_phase()

                # Handle miss in TRUST mode (FTP eviction).
                evicted = self._ftp_evict(exclude=pid)
                self._evict(evicted)
                self._add(pid)
                self._trust_phase_faults += 1

                logger.debug(
                    "t=%d TRUST MISS %s (post-transition) | evicted=%s | faults=%d/%d",
                    request.t, pid, evicted, self._trust_phase_faults, self._trust_budget,
                )

                # Check if the newly reset trust budget is immediately exhausted
                # (edge case: trust_budget could in theory be 0 if
                # initial_trust_budget was set to 0 externally).
                if self._trust_phase_faults >= self._trust_budget:
                    self._start_doubt_phase()

        else:
            # Cache not yet full: just fetch.
            self._add(pid)
            self._marked.add(pid)
            logger.debug(
                "t=%d DOUBT MISS %s (cache not full) | marked=%d",
                request.t, pid, len(self._marked),
            )

        return CacheEvent(t=request.t, page_id=pid, hit=False, cost=cost, evicted=evicted)

    # ------------------------------------------------------------------
    # Mode transitions
    # ------------------------------------------------------------------

    def _start_doubt_phase(self) -> None:
        """Switch from TRUST to DOUBT, initialise a fresh Marker phase."""
        self._mode = "doubt"
        self._marked = set()
        self._trust_phase_faults = 0
        logger.debug("→ Entering DOUBT phase (epoch %d, new budget will be %d)",
                     self._epoch, self._trust_budget * 2)

    def _end_doubt_phase(self) -> None:
        """Switch from DOUBT to TRUST, double the trust budget (epoch doubling).

        INTERPRETATION NOTE B: The trust budget doubles after each completed
        DOUBT phase.  This exponential growth bounds the total overhead of
        DOUBT phases to O(log k) times the cost of an optimal algorithm, even
        in the worst case (all predictions adversarial).

        Concretely: in epoch i, the trust budget is k · 2^i.  If predictions
        are always wrong, a DOUBT phase is triggered every k · 2^i faults.
        Each Marker phase costs at most O(k · log k) faults.  Summing across
        epochs gives the O(log k)-robustness bound.
        """
        self._mode = "trust"
        self._epoch += 1
        self._trust_budget *= 2  # INTERPRETATION NOTE B
        self._trust_phase_faults = 0
        self._marked = set()

    # ------------------------------------------------------------------
    # Eviction rules
    # ------------------------------------------------------------------

    def _ftp_evict(self, exclude: PageId) -> PageId:
        """FTP eviction: evict cached page with largest predicted_next.

        Excludes *exclude* (the page being fetched) from consideration.
        Tie-breaking: alphabetical on page_id for full determinism.

        Used in TRUST mode and on the transition step when DOUBT ends.
        """
        candidates = [q for q in self._cache.current_cache() if q != exclude]
        if not candidates:
            raise RuntimeError(
                f"_ftp_evict: no candidates (cache={self._cache.current_cache()}, "
                f"exclude={exclude})"
            )
        return max(candidates, key=lambda q: (self._predicted_next.get(q, math.inf), q))

    def _marker_evict(self, candidates: list) -> PageId:
        """Marker eviction: evict the LRU page among *candidates*.

        LRU is determined by ``_last_access``: the page with the smallest
        last-access timestamp is evicted first.
        Tie-breaking: alphabetical on page_id.

        Used in DOUBT mode when unmarked pages are available.
        """
        if not candidates:
            raise RuntimeError("_marker_evict: empty candidates list")
        return min(candidates, key=lambda q: (self._last_access.get(q, -1), q))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        """Current operating mode: ``"trust"`` or ``"doubt"``."""
        return self._mode

    @property
    def epoch(self) -> int:
        """Number of completed DOUBT phases (epochs)."""
        return self._epoch

    @property
    def trust_budget(self) -> int:
        """Current trust budget (faults allowed before next DOUBT phase)."""
        return self._trust_budget

    @property
    def trust_phase_faults(self) -> int:
        """Number of faults incurred in the current TRUST phase."""
        return self._trust_phase_faults

    def marked_snapshot(self) -> frozenset:
        """Return an immutable snapshot of pages marked in the current DOUBT phase.

        Returns an empty frozenset if currently in TRUST mode.
        """
        return frozenset(self._marked)

    def predicted_next_snapshot(self) -> Dict[PageId, float]:
        """Return a copy of the current predicted_next mapping."""
        return dict(self._predicted_next)
