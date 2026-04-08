"""Deterministic LA weighted paging (class-level dynamics, faithful-style discretization).

Reference
---------
Bansal, Coester, Kumar, Purohit, Vee.
"Learning-Augmented Weighted Paging." SODA 2022.

====================================================================
STEP-0 PAPER-TO-CODE NOTE (Baseline 1 faithful reimplementation attempt)
====================================================================
1) Data structures for x_i, mu_i, S_i
--------------------------------------
- One ``WeightClassState`` per distinct weight class i.
- ``x`` (float): cache mass assigned to class i.
- ``S`` (IntervalSet): finite union of disjoint half-open intervals [l, r).
- ``mu`` is represented as ``S.measure()``.

2) Grouping pages into classes
------------------------------
Pages are grouped by exact weight values (paper's weight classes).

3) Within-class ranking
-----------------------
Predictions are used only *within each class*: pages are ordered by the most
recent predicted next-arrival (ascending; ties by page id).  Requested page
rank q is 1-indexed within its class ordering.

4) Continuous process discretization
------------------------------------
The paper is continuous in pointer p.  We implement an explicit small-step,
event-style integrator over p from q-1 to q:
- p increases at speed 8,
- no dynamics while p <= x_r,
- while p > x_r, update x_i and mu_i via the paper ODE right-hand sides,
  then apply set updates to S_i.

INTERPRETATION NOTE: This is a numerical discretization (Euler-style) of the
continuous dynamics, not an exact symbolic solver.

5) Dummy-page / boundary handling
---------------------------------
We model the paper's "+ell dummy pages" boundary trick by enforcing x_i >= 1
for every class and total mass sum_i x_i = k + ell in the internal dynamics.
Real cache slots remain k; the extra ell mass is interpreted as per-class dummy
mass that is never mapped to real pages.

6) Old la_det policy status
---------------------------
The pre-existing ``la_det`` policy is retained as an *approximate heuristic*
(weight-normalized prediction score), and this module introduces a separate
policy name ``la_det_faithful`` for the class-level process.

7) Reused pieces
----------------
The old policy's weight-class grouping and prediction bookkeeping ideas are
reused; global page-level eviction scoring is not reused.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


EPS = 1e-9


@dataclass
class IntervalSet:
    intervals: List[Tuple[float, float]] = field(default_factory=list)

    def normalize(self) -> None:
        xs = [(l, r) for l, r in self.intervals if r - l > EPS]
        if not xs:
            self.intervals = []
            return
        xs.sort()
        out: List[Tuple[float, float]] = [xs[0]]
        for l, r in xs[1:]:
            pl, pr = out[-1]
            if l <= pr + EPS:
                out[-1] = (pl, max(pr, r))
            else:
                out.append((l, r))
        self.intervals = out

    def measure(self) -> float:
        return sum(r - l for l, r in self.intervals)

    def contains(self, x: float) -> bool:
        return any(l - EPS <= x < r + EPS for l, r in self.intervals)

    def add_interval(self, l: float, r: float) -> None:
        if r - l <= EPS:
            return
        self.intervals.append((l, r))
        self.normalize()

    def remove_interval(self, l: float, r: float) -> None:
        if r - l <= EPS:
            return
        out: List[Tuple[float, float]] = []
        for a, b in self.intervals:
            if b <= l + EPS or a >= r - EPS:
                out.append((a, b))
            else:
                if a < l - EPS:
                    out.append((a, l))
                if b > r + EPS:
                    out.append((r, b))
        self.intervals = out
        self.normalize()

    def remove_left(self, length: float) -> None:
        rem = length
        out: List[Tuple[float, float]] = []
        for l, r in self.intervals:
            if rem <= EPS:
                out.append((l, r))
                continue
            span = r - l
            if span <= rem + EPS:
                rem -= span
            else:
                out.append((l + rem, r))
                rem = 0.0
        self.intervals = out

    def remove_right(self, length: float) -> None:
        rem = length
        out: List[Tuple[float, float]] = []
        for l, r in reversed(self.intervals):
            if rem <= EPS:
                out.append((l, r))
                continue
            span = r - l
            if span <= rem + EPS:
                rem -= span
            else:
                out.append((l, r - rem))
                rem = 0.0
        out.reverse()
        self.intervals = out


@dataclass
class WeightClassState:
    weight: float
    page_ids: Set[PageId]
    x: float = 0.0
    s: IntervalSet = field(default_factory=IntervalSet)


class LAWeightedPagingDeterministicFaithful(BasePolicy):
    """Class-level deterministic LA weighted paging with faithful-style dynamics."""

    name = "la_weighted_paging_det_faithful"

    def __init__(self, step_dp: float = 0.02) -> None:
        self._step_dp = step_dp

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        self._predicted_next: Dict[PageId, float] = {pid: math.inf for pid in pages}
        by_w: Dict[float, Set[PageId]] = {}
        for pid, p in pages.items():
            by_w.setdefault(p.weight, set()).add(pid)

        self._classes: List[WeightClassState] = [
            WeightClassState(weight=w, page_ids=ids)
            for w, ids in sorted(by_w.items(), key=lambda kv: kv[0])
        ]
        self._class_idx: Dict[float, int] = {c.weight: i for i, c in enumerate(self._classes)}
        self._page_to_class: Dict[PageId, int] = {}
        for i, c in enumerate(self._classes):
            for pid in c.page_ids:
                self._page_to_class[pid] = i

        ell = len(self._classes)
        self._ell = ell
        self._k_eff = self._cache.capacity + ell
        base = self._k_eff / ell
        for c in self._classes:
            c.x = base
            c.s = IntervalSet([])

        self._last_touch: Dict[PageId, int] = {pid: -10**9 for pid in pages}

    # ------------------------ ranking helpers ------------------------
    def _class_ranking(self, ci: int) -> List[PageId]:
        c = self._classes[ci]
        return sorted(c.page_ids, key=lambda pid: (self._predicted_next.get(pid, math.inf), pid))

    def _rank_of(self, ci: int, pid: PageId) -> int:
        order = self._class_ranking(ci)
        return order.index(pid) + 1

    # ----------------------- dynamics helpers ------------------------
    def _mu(self, i: int) -> float:
        return self._classes[i].s.measure()

    def _mass_repair(self) -> None:
        # Enforce x_i >= 1 and sum x_i = k+ell.
        xs = [max(1.0, c.x) for c in self._classes]
        total = sum(xs)
        if total <= EPS:
            return
        target = self._k_eff
        # distribute correction over classes above floor when possible
        diff = target - total
        if abs(diff) <= 1e-8:
            for c, v in zip(self._classes, xs):
                c.x = v
            return
        free = [i for i, v in enumerate(xs) if v > 1.0 + EPS]
        if not free:
            for c, v in zip(self._classes, xs):
                c.x = v
            return
        share = diff / len(free)
        for i in free:
            xs[i] = max(1.0, xs[i] + share)
        # final normalize
        scale_total = sum(xs)
        if scale_total > EPS:
            adjust = (target - scale_total) / len(xs)
            xs = [max(1.0, v + adjust) for v in xs]
        for c, v in zip(self._classes, xs):
            c.x = v

    def _integrate_request_class(self, r: int, q: int) -> None:
        delta = 1.0 / self._ell
        p = q - 1.0
        p_end = float(q)

        while p < p_end - EPS:
            step = min(self._step_dp, p_end - p)
            p_next = p + step

            xr = self._classes[r].x
            if p_next <= xr + EPS:
                p = p_next
                continue

            # active duration when pointer is to the right of x_r
            active = p_next - max(p, xr)
            if active <= EPS:
                p = p_next
                continue

            mus = [max(self._mu(i), EPS) for i in range(self._ell)]
            M = sum(mus)
            betas = [
                (mus[i] + delta * M) / (self._classes[i].weight * M)
                for i in range(self._ell)
            ]
            B = sum(betas)

            old_x = [c.x for c in self._classes]
            xdot = [(-betas[i] / B) for i in range(self._ell)]
            xdot[r] += 1.0

            for i in range(self._ell):
                self._classes[i].x += xdot[i] * active
            self._mass_repair()

            # S updates for i != r when x_i decreased.
            for i in range(self._ell):
                if i == r:
                    continue
                if self._classes[i].x < old_x[i] - EPS:
                    self._classes[i].s.add_interval(self._classes[i].x, old_x[i])

            # Requested class S_r updates.
            sr = self._classes[r].s
            xrdot = xdot[r]
            if xrdot > EPS:
                sr.remove_left(xrdot * active)
            sr.remove_right(active)

            # If p not in S_r, add uncovered mass from (q-1, p] at rate 2.
            if not sr.contains(p):
                sr.add_interval(max(q - 1.0, p), min(p_end, p + 2.0 * active))

            # Keep S_i in [x_i, +inf)
            for i in range(self._ell):
                xi = self._classes[i].x
                self._classes[i].s.remove_interval(-1e18, xi)

            p = p_next

    # ----------------------- cache mapping ---------------------------
    def _target_real_counts(self) -> Dict[int, int]:
        # remove dummy mass 1 from each class
        real_mass = [max(0.0, c.x - 1.0) for c in self._classes]
        floors = [int(math.floor(v)) for v in real_mass]
        used = sum(floors)
        rem = self._cache.capacity - used
        frac = sorted(
            [(real_mass[i] - floors[i], i) for i in range(self._ell)],
            reverse=True,
        )
        counts = floors[:]
        for _, i in frac:
            if rem <= 0:
                break
            counts[i] += 1
            rem -= 1
        return {i: counts[i] for i in range(self._ell)}

    def _target_cache_set(self) -> Set[PageId]:
        counts = self._target_real_counts()
        target: Set[PageId] = set()
        for i in range(self._ell):
            order = self._class_ranking(i)
            target.update(order[: counts[i]])
        return target

    def _evict_for_target(self, target: Set[PageId]) -> Optional[PageId]:
        if not self._cache.is_full():
            return None
        candidates = set(self._cache.current_cache()) - target
        if not candidates:
            candidates = set(self._cache.current_cache())
        victim = min(candidates, key=lambda p: (self._last_touch.get(p, -10**9), p))
        self._evict(victim)
        return victim

    def _class_snapshot(self) -> Dict[str, Dict[str, object]]:
        out: Dict[str, Dict[str, object]] = {}
        for i, c in enumerate(self._classes):
            out[str(c.weight)] = {
                "class_index": i,
                "x": c.x,
                "mu": c.s.measure(),
                "intervals": [(round(l, 5), round(r, 5)) for l, r in c.s.intervals],
            }
        return out

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id
        self._predicted_next[pid] = request.predicted_next
        self._last_touch[pid] = request.t

        r = self._page_to_class[pid]
        q = self._rank_of(r, pid)

        self._integrate_request_class(r, q)
        target = self._target_cache_set()

        hit = self.in_cache(pid)
        evicted = None
        if hit:
            self._record_hit()
            cost = 0.0
        else:
            cost = self._pages[pid].weight
            self._record_miss(cost)
            evicted = self._evict_for_target(target)
            self._add(pid)

        return CacheEvent(
            t=request.t,
            page_id=pid,
            hit=hit,
            cost=cost,
            evicted=evicted,
            diagnostics={
                "requested_class": r,
                "requested_rank": q,
                "x_mu": self._class_snapshot(),
                "dummy_boundary_active": True,
            },
        )
