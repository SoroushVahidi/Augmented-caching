"""Offline general caching baseline via LP relaxation + deterministic rounding.

This solver targets the offline file caching/general caching setting with:
- arbitrary positive sizes,
- arbitrary positive retrieval costs,
- known full request sequence,
- optional insertion (bypass) on misses.

Method classification
---------------------
This implementation is an *approximation-inspired* baseline (not a theorem-faithful
implementation of the full Bar-Noy et al. algorithmic framework).

It builds a weighted interval-packing LP relaxation over reuse intervals and rounds
it into a feasible schedule by deterministic capacity-aware selection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List

import pulp

from lafc.offline.types import OfflineDecision, OfflineSimulationResult
from lafc.types import Page, PageId, Request


@dataclass(frozen=True)
class _ReuseInterval:
    idx: int
    page_id: PageId
    start_t: int
    end_t: int
    size: float
    value: float


@dataclass
class GeneralCachingLPApproxSolver:
    """LP-based offline baseline for general caching."""

    name: str = "offline_general_caching_lp_round"

    def solve(
        self,
        requests: Iterable[Request],
        pages: Dict[PageId, Page],
        capacity: float,
        *,
        page_sizes: Dict[PageId, float],
        allow_bypass: bool = True,
        initial_cache: List[PageId] | None = None,
    ) -> OfflineSimulationResult:
        req_list = list(requests)
        if not req_list:
            raise ValueError("General caching offline solver requires non-empty requests.")
        if capacity <= 0:
            raise ValueError(f"Cache capacity must be > 0, got {capacity}")
        if not allow_bypass:
            raise ValueError("This baseline currently requires allow_bypass=True.")

        self._validate_inputs(req_list, pages, page_sizes, capacity)

        t0 = time.perf_counter()
        initial_cache = list(initial_cache or [])
        bad_initial = [pid for pid in initial_cache if pid not in page_sizes]
        if bad_initial:
            raise ValueError(f"initial_cache contains unknown page_ids: {sorted(set(bad_initial))}")
        initial_total = sum(page_sizes[pid] for pid in initial_cache)
        if initial_total > capacity + 1e-9:
            raise ValueError(
                f"initial_cache total size {initial_total} exceeds capacity {capacity}"
            )
        intervals = self._build_reuse_intervals(req_list, pages, page_sizes, initial_cache)
        lp_meta = self._solve_lp(intervals, len(req_list), capacity)
        selected_ids, rounded_meta = self._round_solution(
            intervals,
            lp_meta["x_values"],
            len(req_list),
            capacity,
        )
        sim = self._simulate_from_selection(
            req_list,
            pages,
            page_sizes,
            capacity,
            intervals,
            selected_ids,
            initial_cache,
        )
        runtime_sec = time.perf_counter() - t0

        sim.diagnostics.update(
            {
                "solver_classification": "approximation-inspired",
                "inspiration": [
                    "Bar-Noy et al. (JACM 2001): resource-allocation style interval packing",
                    "General caching/file caching approximation literature",
                ],
                "model": {
                    "offline_full_sequence_known": True,
                    "optional_insertion_supported": True,
                    "arbitrary_sizes_supported": True,
                    "arbitrary_retrieval_costs_supported": True,
                    "capacity_type": "real-valued positive scalar",
                    "initial_cache_supported": True,
                },
                "lp": lp_meta,
                "rounding": rounded_meta,
                "runtime_sec": runtime_sec,
            }
        )
        return sim

    def _validate_inputs(
        self,
        requests: List[Request],
        pages: Dict[PageId, Page],
        page_sizes: Dict[PageId, float],
        capacity: float,
    ) -> None:
        missing_pages = sorted({r.page_id for r in requests if r.page_id not in pages})
        if missing_pages:
            raise ValueError(f"Missing page costs for page_ids: {missing_pages}")

        missing_sizes = sorted({r.page_id for r in requests if r.page_id not in page_sizes})
        if missing_sizes:
            raise ValueError(
                "Missing page sizes for page_ids: "
                f"{missing_sizes}. Provide a sizes map in the trace."
            )

        bad_sizes = {pid: s for pid, s in page_sizes.items() if s <= 0}
        if bad_sizes:
            raise ValueError(f"All page sizes must be > 0, got: {bad_sizes}")

        bad_costs = {pid: p.weight for pid, p in pages.items() if p.weight <= 0}
        if bad_costs:
            raise ValueError(f"All retrieval costs must be > 0, got: {bad_costs}")

        fits = any(page_sizes[r.page_id] <= capacity for r in requests)
        if not fits:
            raise ValueError(
                "No requested page fits in capacity; problem is infeasible under this model."
            )

    def _build_reuse_intervals(
        self,
        requests: List[Request],
        pages: Dict[PageId, Page],
        page_sizes: Dict[PageId, float],
        initial_cache: List[PageId],
    ) -> List[_ReuseInterval]:
        positions: Dict[PageId, List[int]] = {}
        for t, req in enumerate(requests):
            positions.setdefault(req.page_id, []).append(t)

        intervals: List[_ReuseInterval] = []
        idx = 0
        for pid in sorted(set(initial_cache)):
            first_occ = positions.get(pid, [])
            if first_occ:
                intervals.append(
                    _ReuseInterval(
                        idx=idx,
                        page_id=pid,
                        start_t=-1,
                        end_t=first_occ[0],
                        size=float(page_sizes[pid]),
                        value=float(pages[pid].weight),
                    )
                )
                idx += 1
        for pid, occ in positions.items():
            for i in range(len(occ) - 1):
                intervals.append(
                    _ReuseInterval(
                        idx=idx,
                        page_id=pid,
                        start_t=occ[i],
                        end_t=occ[i + 1],
                        size=float(page_sizes[pid]),
                        value=float(pages[pid].weight),
                    )
                )
                idx += 1
        return intervals

    def _solve_lp(
        self,
        intervals: List[_ReuseInterval],
        n_requests: int,
        capacity: float,
    ) -> Dict[str, object]:
        if not intervals:
            return {
                "status": "no_intervals",
                "num_intervals": 0,
                "num_capacity_constraints": max(n_requests - 1, 0),
                "objective_value": 0.0,
                "x_values": {},
            }

        model = pulp.LpProblem("general_caching_interval_lp", pulp.LpMaximize)
        x = {
            it.idx: pulp.LpVariable(f"x_{it.idx}", lowBound=0.0, upBound=1.0)
            for it in intervals
        }

        model += pulp.lpSum(it.value * x[it.idx] for it in intervals)

        for slot in range(0, n_requests):
            covering = [it for it in intervals if it.start_t < slot <= it.end_t]
            if covering:
                model += (
                    pulp.lpSum(it.size * x[it.idx] for it in covering) <= capacity,
                    f"cap_slot_{slot}",
                )

        solver = pulp.PULP_CBC_CMD(msg=False)
        model.solve(solver)

        x_vals = {it.idx: float(x[it.idx].value() or 0.0) for it in intervals}
        return {
            "status": pulp.LpStatus[model.status],
            "num_intervals": len(intervals),
            "num_capacity_constraints": max(n_requests - 1, 0),
            "objective_value": float(pulp.value(model.objective) or 0.0),
            "x_values": x_vals,
        }

    def _round_solution(
        self,
        intervals: List[_ReuseInterval],
        x_values: Dict[int, float],
        n_requests: int,
        capacity: float,
    ) -> tuple[set[int], Dict[str, object]]:
        residual = {slot: float(capacity) for slot in range(0, n_requests)}

        feasible_intervals = [it for it in intervals if it.size <= capacity]

        ranked = sorted(
            feasible_intervals,
            key=lambda it: (
                x_values.get(it.idx, 0.0) * (it.value / it.size),
                x_values.get(it.idx, 0.0),
                it.value / it.size,
                -it.start_t,
                it.page_id,
            ),
            reverse=True,
        )

        selected: set[int] = set()
        rounded_obj = 0.0
        for it in ranked:
            if x_values.get(it.idx, 0.0) <= 0.0:
                continue
            slots = range(it.start_t + 1, it.end_t + 1)
            if all(residual[s] + 1e-12 >= it.size for s in slots):
                selected.add(it.idx)
                rounded_obj += it.value
                for s in slots:
                    residual[s] -= it.size

        return selected, {
            "strategy": "deterministic_capacity_aware_greedy_from_lp",
            "selected_intervals": len(selected),
            "rounded_saved_cost": rounded_obj,
        }

    def _simulate_from_selection(
        self,
        requests: List[Request],
        pages: Dict[PageId, Page],
        page_sizes: Dict[PageId, float],
        capacity: float,
        intervals: List[_ReuseInterval],
        selected_ids: set[int],
        initial_cache: List[PageId],
    ) -> OfflineSimulationResult:
        selected_by_start: Dict[int, _ReuseInterval] = {}
        for it in intervals:
            if it.idx in selected_ids and it.start_t >= 0:
                selected_by_start[it.start_t] = it
        selected_initial = {
            it.page_id: it for it in intervals if it.idx in selected_ids and it.start_t < 0
        }

        cache_until: Dict[PageId, int] = {}
        for pid in initial_cache:
            if pid in selected_initial:
                cache_until[pid] = selected_initial[pid].end_t
            else:
                cache_until[pid] = 0
        decisions: List[OfflineDecision] = []
        misses = 0
        hits = 0
        insertions = 0
        bypasses = 0
        total_cost = 0.0

        for t, req in enumerate(requests):
            for pid in list(cache_until.keys()):
                if cache_until[pid] < t:
                    del cache_until[pid]

            pid = req.page_id
            hit = pid in cache_until
            inserted = False
            bypassed = False

            if hit:
                cost = 0.0
                hits += 1
            else:
                cost = float(pages[pid].weight)
                misses += 1
                total_cost += cost

                selected = selected_by_start.get(t)
                if selected is not None and selected.page_id == pid:
                    if page_sizes[pid] <= capacity:
                        current_occ = sum(page_sizes[q] for q in cache_until)
                        needed = current_occ + page_sizes[pid] - capacity
                        if needed > 1e-12:
                            removable = sorted(
                                [q for q in cache_until.keys() if q != pid],
                                key=lambda q: (cache_until[q], q),
                            )
                            for q in removable:
                                del cache_until[q]
                                needed -= page_sizes[q]
                                if needed <= 1e-12:
                                    break
                        cache_until[pid] = selected.end_t
                        inserted = True
                        insertions += 1
                    else:
                        bypassed = True
                        bypasses += 1
                else:
                    bypassed = True
                    bypasses += 1

            selected_now = selected_by_start.get(t)
            if hit and selected_now is not None and selected_now.page_id == pid:
                cache_until[pid] = selected_now.end_t

            if hit and (selected_now is None or selected_now.page_id != pid):
                if cache_until.get(pid, -1) == t:
                    del cache_until[pid]

            occupancy = sum(page_sizes[q] for q in cache_until)
            if occupancy > capacity + 1e-9:
                raise RuntimeError(
                    f"Rounded schedule violated capacity at t={t}: {occupancy} > {capacity}"
                )

            decisions.append(
                OfflineDecision(
                    t=t,
                    page_id=pid,
                    hit=hit,
                    cost=cost,
                    inserted=inserted,
                    bypassed=bypassed,
                    cache_occupancy=occupancy,
                )
            )

        return OfflineSimulationResult(
            solver_name=self.name,
            capacity=int(capacity) if float(capacity).is_integer() else capacity,
            total_requests=len(requests),
            total_hits=hits,
            total_misses=misses,
            decisions=decisions,
            diagnostics={
                "total_retrieval_cost": total_cost,
                "insertions": insertions,
                "bypasses": bypasses,
            },
        )
