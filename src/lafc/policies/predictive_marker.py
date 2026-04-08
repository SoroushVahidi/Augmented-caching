"""
Predictive Marker caching algorithm (unweighted paging).

Reference
---------
Lykouris, Vassilvitskii.
"Competitive Caching with Machine Learned Advice."
ICML 2018 / JACM 2021.

============================================================
PAPER-TO-CODE IMPLEMENTATION NOTE
============================================================

1. Exact paging setting
-----------------------
Standard paging (unweighted caching):
- Cache capacity k, all pages unit size.
- All misses have unit cost (cost = 1).
- Request sequence σ_1, ..., σ_T.
- Each request σ_t comes with a prediction τ_t: the predicted next
  arrival time of σ_t after time t.  τ_t = ∞ means "never again."
- Objective: minimise total cache misses.

2. Prediction model
-------------------
For each request at time t:
  - requested page: σ_t
  - predicted next arrival: τ_t ∈ {t+1, ..., T, ∞}
  - actual next arrival: a_t (computed offline from the trace, used
    only for diagnostics — NOT used by the online algorithm)
Prediction error: η = Σ_t |τ_t − a_t|

3. Phase structure of Marker (inherited by Predictive Marker)
-------------------------------------------------------------
A phase is a maximal time interval during which at most k distinct
pages are faulted on.  In implementation terms:

- At the start of each phase all k cached pages are "unmarked."
- When page p is requested:
    HIT  (p ∈ cache):  mark p.
    MISS (p ∉ cache):
        If all cached pages are marked (|M| = k):
            Start new phase: unmark all (M ← ∅).
        Evict a page v from the unmarked set (C ∖ M).
        Fetch p and mark it.

4. How Predictive Marker modifies Marker
-----------------------------------------
Identical phase structure; the only change is the eviction rule:

    Marker:            evict arbitrary v ∈ C ∖ M
    Predictive Marker: evict v* = argmax_{v ∈ C ∖ M} predicted_next[v]

That is, among all currently unmarked cached pages, evict the one
whose most recently received predicted next arrival is farthest in
the future.

5. Clean chains (diagnostics)
------------------------------
A clean chain is a concept from the paper's *analysis* (Section 4,
Proof of Theorem 2), used to bound the number of extra faults by η.

Informally, an eviction step is "clean" if the victim chosen by the
algorithm (argmax predicted_next among unmarked) coincides with the
page that would have been evicted by an offline algorithm knowing the
*actual* next arrivals (argmax actual_next among unmarked).

A phase is clean if all evictions within it are individually clean.
A clean chain is a maximal consecutive sequence of clean phases.

In code, clean-chain tracking is a **post-hoc diagnostic** that uses
actual_next (available in the Request object but NOT used for
eviction decisions).  It is computed during the simulation and stored
in SimulationResult.extra_diagnostics.

6. Quantities tracked in code
-------------------------------
  _marked         : set[PageId]
      Marked pages in the current phase (subset of _cache).
  _phase          : int
      Current phase index (1-based).
  _predicted_next : Dict[PageId, float]
      Most recently received prediction for each page.
      Initialised to math.inf (unknown → treat as "never needed").
  _actual_next    : Dict[PageId, float]
      Most recently received actual_next for each page.
      Used only for clean-chain diagnostics (not for eviction).
  _phase_log      : list[PhaseRecord]
      One entry per completed (or in-progress) phase.

7. Exact vs interpreted parts
-------------------------------
EXACT (from the paper):
  - Phase structure (Section 3).
  - Unit cost model.
  - Eviction rule: argmax predicted_next among unmarked pages.

INTERPRETATION NOTE A — Default prediction
  The paper does not specify a default for pages with no prior
  prediction.  We use math.inf (treat as "never needed again"):
  such pages are eviction candidates, but are deprioritised over
  pages with explicit finite predictions.

INTERPRETATION NOTE B — Tie-breaking in eviction
  When two unmarked pages share the same predicted_next, the paper
  does not specify a tie-breaking rule.  We break ties deterministically
  by evicting the lexicographically largest page_id.  This ensures
  reproducibility and does not affect theoretical guarantees.

INTERPRETATION NOTE C — Expired predictions
  If a page's stored predicted_next < current_t (the predicted arrival
  has passed without the page being requested), we use the stored value
  as-is.  An expired prediction still serves as an ordering signal;
  resetting to ∞ would be an alternative interpretation not mentioned
  in the paper.

INTERPRETATION NOTE D — Clean-phase definition
  The paper uses "clean chains" in its analysis but does not give an
  explicit per-step algorithmic definition.  We define a phase as
  clean if every eviction within it satisfies:
    evicted page = argmax actual_next among unmarked pages at that moment.
  This is the natural formalisation consistent with the paper's proof
  sketch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set

from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


# ---------------------------------------------------------------------------
# Diagnostic dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EvictionRecord:
    """Diagnostic information for a single eviction within a phase."""

    t: int
    victim: PageId
    predicted_next_victim: float  # stored predicted_next of the evicted page
    actual_next_victim: float     # actual_next of the evicted page (diagnostic only)
    oracle_victim: PageId         # argmax actual_next among unmarked (offline oracle)
    is_clean: bool                # True iff victim == oracle_victim


@dataclass
class PhaseRecord:
    """Summary of a single phase (for clean-chain diagnostics)."""

    phase_id: int
    start_t: int
    evictions: List[EvictionRecord] = field(default_factory=list)
    end_t: Optional[int] = None   # Filled in when the phase ends or at simulation end.
    is_clean: bool = True         # Set to False if any eviction is dirty.


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------


class PredictiveMarkerPolicy(BasePolicy):
    """Predictive Marker caching policy (unweighted).

    Faithfully implements the Predictive Marker algorithm from:
        Lykouris & Vassilvitskii, "Competitive Caching with Machine
        Learned Advice," ICML 2018 / JACM 2021.

    Phase-based algorithm identical to Marker except that on a fault the
    evicted page is argmax_{v ∈ unmarked} predicted_next[v] (the unmarked
    page whose predicted next arrival is farthest away).

    All costs are unit (1.0 per miss); page weights are ignored.

    Additionally tracks clean-chain diagnostics (stored in
    SimulationResult.extra_diagnostics by the runner after the run).
    """

    name: str = "predictive_marker"

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        self._marked: Set[PageId] = set()
        self._phase: int = 1

        # INTERPRETATION NOTE A: default = math.inf.
        self._predicted_next: Dict[PageId, float] = {}

        # Actual next arrivals — updated from Request.actual_next on each
        # request; used only for clean-chain diagnostics (not for eviction).
        self._actual_next: Dict[PageId, float] = {}

        # Diagnostic log: one PhaseRecord per phase (the last may be open).
        self._phase_log: List[PhaseRecord] = [PhaseRecord(phase_id=1, start_t=0)]

    # ------------------------------------------------------------------
    # Main algorithm step
    # ------------------------------------------------------------------

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id
        evicted: Optional[PageId] = None

        # Always update predictions for the requested page (hit or miss).
        self._predicted_next[pid] = request.predicted_next
        # Track actual_next for diagnostics only (not used in eviction).
        self._actual_next[pid] = request.actual_next

        if self.in_cache(pid):
            # Cache hit: mark the page (idempotent).
            self._marked.add(pid)
            self._record_hit()
            return CacheEvent(
                t=request.t,
                page_id=pid,
                hit=True,
                cost=0.0,
                phase=self._phase,
            )

        # Cache miss — unit cost.
        self._record_miss(1.0)

        if self._cache.is_full():
            if len(self._marked) == self._cache.capacity:
                # All k pages are marked → start a new phase.
                # Close the current phase record.
                self._phase_log[-1].end_t = request.t - 1
                self._phase += 1
                self._marked = set()
                self._phase_log.append(
                    PhaseRecord(phase_id=self._phase, start_t=request.t)
                )

            # Evict the unmarked page with the largest predicted next arrival.
            evicted = self._choose_victim(request.t)
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

    def _choose_victim(self, current_t: int) -> PageId:
        """Evict the unmarked page with the largest predicted next arrival.

        INTERPRETATION NOTE B: ties broken deterministically by
        lexicographically largest page_id.

        Also records an EvictionRecord in the current phase log entry,
        comparing the chosen victim against the oracle victim (argmax
        actual_next).  actual_next is NOT used for the eviction decision.
        """
        unmarked: FrozenSet[PageId] = self._cache.current_cache() - self._marked
        if not unmarked:
            raise RuntimeError(
                "No unmarked page available to evict — this indicates a bug. "
                f"cache={self._cache.current_cache()}, marked={self._marked}"
            )

        # Algorithm eviction: argmax predicted_next (INTERPRETATION NOTE B for ties).
        victim = max(
            unmarked,
            key=lambda q: (self._predicted_next.get(q, math.inf), q),
        )

        # Oracle eviction: argmax actual_next (for diagnostics only).
        oracle_victim = max(
            unmarked,
            key=lambda q: (self._actual_next.get(q, math.inf), q),
        )

        eviction_rec = EvictionRecord(
            t=current_t,
            victim=victim,
            predicted_next_victim=self._predicted_next.get(victim, math.inf),
            actual_next_victim=self._actual_next.get(victim, math.inf),
            oracle_victim=oracle_victim,
            is_clean=(victim == oracle_victim),
        )
        self._phase_log[-1].evictions.append(eviction_rec)
        if not eviction_rec.is_clean:
            self._phase_log[-1].is_clean = False

        return victim

    # ------------------------------------------------------------------
    # Post-simulation diagnostics
    # ------------------------------------------------------------------

    def close_final_phase(self, last_t: int) -> None:
        """Mark the last phase as ended at *last_t*.

        Called by the runner after the last request to close the open
        PhaseRecord.
        """
        if self._phase_log and self._phase_log[-1].end_t is None:
            self._phase_log[-1].end_t = last_t

    def compute_clean_chains(self) -> Dict[str, Any]:
        """Compute clean-chain diagnostics from the phase log.

        Returns a dict with:
        ``"phases"``
            List of per-phase dicts: phase_id, start_t, end_t,
            num_evictions, is_clean.
        ``"num_clean_phases"``
            Count of phases where every eviction matched the oracle.
        ``"num_dirty_phases"``
            Count of phases with at least one dirty eviction.
        ``"clean_chains"``
            List of ``[start_phase_id, end_phase_id]`` for each maximal
            consecutive run of clean phases.
        ``"num_clean_chains"``
            Number of distinct clean chains.
        ``"total_clean_evictions"``
            Evictions where victim == oracle_victim.
        ``"total_dirty_evictions"``
            Evictions where victim != oracle_victim.
        """
        phases_out: List[Dict[str, Any]] = []
        clean_phase_ids: List[int] = []

        for pr in self._phase_log:
            phases_out.append(
                {
                    "phase_id": pr.phase_id,
                    "start_t": pr.start_t,
                    "end_t": pr.end_t,
                    "num_evictions": len(pr.evictions),
                    "is_clean": pr.is_clean,
                }
            )
            if pr.is_clean:
                clean_phase_ids.append(pr.phase_id)

        # Find maximal consecutive runs of clean phases.
        clean_chains: List[List[int]] = []
        if clean_phase_ids:
            chain_start = clean_phase_ids[0]
            chain_prev = clean_phase_ids[0]
            for pid in clean_phase_ids[1:]:
                if pid == chain_prev + 1:
                    chain_prev = pid
                else:
                    clean_chains.append([chain_start, chain_prev])
                    chain_start = pid
                    chain_prev = pid
            clean_chains.append([chain_start, chain_prev])

        total_clean_ev = sum(
            1 for pr in self._phase_log for ev in pr.evictions if ev.is_clean
        )
        total_dirty_ev = sum(
            1 for pr in self._phase_log for ev in pr.evictions if not ev.is_clean
        )
        num_clean = sum(1 for pr in self._phase_log if pr.is_clean)
        num_dirty = sum(1 for pr in self._phase_log if not pr.is_clean)

        return {
            "phases": phases_out,
            "num_clean_phases": num_clean,
            "num_dirty_phases": num_dirty,
            "clean_chains": clean_chains,
            "num_clean_chains": len(clean_chains),
            "total_clean_evictions": total_clean_ev,
            "total_dirty_evictions": total_dirty_ev,
        }

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def current_phase(self) -> int:
        """Return the current phase index (1-based)."""
        return self._phase

    def marked_pages(self) -> frozenset:
        """Return a snapshot of the currently marked page ids."""
        return frozenset(self._marked)

    def predicted_next_snapshot(self) -> Dict[PageId, float]:
        """Return a copy of the current predicted_next mapping."""
        return dict(self._predicted_next)
