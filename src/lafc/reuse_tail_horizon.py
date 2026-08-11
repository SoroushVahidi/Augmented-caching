"""Reuse-tail / horizon-exceedance diagnostic primitives.

This module measures future request-position delay for resident eviction
candidates.  It deliberately does *not* compute classical reuse distance or
stack distance, which count distinct intervening objects.  Here:

    T = next_request_index(candidate) - decision_request_index

for a candidate object already resident at a full-cache miss decision.  If the
candidate is never requested again in the remaining trace, T is infinite.
"""

from __future__ import annotations

import bisect
import collections
import hashlib
import math
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Sequence

from lafc.supervision_objective_ablation import _build_occurrence_index
from lafc.types import PageId, Request


DEFAULT_HORIZONS = (1, 2, 4, 8, 16)
PRIMARY_HORIZON = 4
QUANTILE_LEVELS = (0.50, 0.75, 0.90, 0.95, 0.99)


@dataclass(frozen=True)
class ReuseTailObservation:
    family: str
    trace_name: str
    capacity: int
    decision_index: int
    decision_id: str
    candidate_page_id: PageId
    next_reuse_request_index: Optional[int]
    t: float

    @property
    def never_reused(self) -> bool:
        return self.next_reuse_request_index is None


@dataclass
class CellAccumulator:
    family: str
    trace_name: str
    capacity: int
    horizons: Sequence[int]
    decision_points: int = 0
    observations: int = 0
    finite_reuse_count: int = 0
    never_reused_count: int = 0
    finite_t_values: List[int] = None  # type: ignore[assignment]
    exceed_counts: Dict[int, int] = None  # type: ignore[assignment]
    finite_exceed_counts: Dict[int, int] = None  # type: ignore[assignment]
    within_counts: Dict[int, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.finite_t_values is None:
            self.finite_t_values = []
        if self.exceed_counts is None:
            self.exceed_counts = {int(h): 0 for h in self.horizons}
        if self.finite_exceed_counts is None:
            self.finite_exceed_counts = {int(h): 0 for h in self.horizons}
        if self.within_counts is None:
            self.within_counts = {int(h): 0 for h in self.horizons}

    def record(self, obs: ReuseTailObservation) -> None:
        self.observations += 1
        if obs.never_reused:
            self.never_reused_count += 1
            for h in self.horizons:
                self.exceed_counts[int(h)] += 1
            return

        t_value = int(obs.t)
        self.finite_reuse_count += 1
        self.finite_t_values.append(t_value)
        for h in self.horizons:
            h = int(h)
            if t_value > h:
                self.exceed_counts[h] += 1
                self.finite_exceed_counts[h] += 1
            else:
                self.within_counts[h] += 1


def next_reuse_request_index(
    occurrence_index: Mapping[PageId, Sequence[int]],
    candidate: PageId,
    decision_index: int,
) -> Optional[int]:
    """Return the next absolute request index for candidate after decision.

    The decision index is the incoming missed request position.  Resident
    candidates differ from that incoming object, so the next reuse is strictly
    after the decision point.  The returned T is therefore measured in future
    request positions, not distinct intervening objects.
    """
    positions = occurrence_index.get(candidate, ())
    pos = bisect.bisect_right(positions, decision_index)
    if pos >= len(positions):
        return None
    return int(positions[pos])


def reuse_delay_t(next_reuse_index: Optional[int], decision_index: int) -> float:
    if next_reuse_index is None:
        return math.inf
    if next_reuse_index <= decision_index:
        raise ValueError(
            f"next_reuse_index must be after decision_index: "
            f"{next_reuse_index} <= {decision_index}"
        )
    return float(next_reuse_index - decision_index)


def stable_candidate_key(family: str, trace_name: str, candidate: PageId) -> str:
    payload = f"{family}|{trace_name}|{candidate}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def iter_resident_candidate_observations(
    requests: Sequence[Request],
    *,
    family: str,
    trace_name: str,
    capacity: int,
    score_start: int,
    score_end: int,
) -> Iterable[ReuseTailObservation]:
    """Yield resident candidates at LRU full-cache miss decisions.

    This mirrors the decision/candidate population in
    `iter_multi_label_candidate_rows`: replay an LRU cache state, emit a
    decision only on a miss when the cache is already full, enumerate every
    resident candidate before inserting the incoming missed object, then evict
    the LRU resident to advance the reference state.
    """
    if capacity <= 0:
        raise ValueError(f"capacity must be positive, got {capacity}")
    if not (0 <= score_start <= score_end <= len(requests)):
        raise ValueError(
            f"invalid score window [{score_start}, {score_end}) for "
            f"{len(requests)} requests"
        )

    occurrence_index = _build_occurrence_index(requests)
    order: "collections.OrderedDict[PageId, None]" = collections.OrderedDict()
    recent_req_hist: Deque[PageId] = collections.deque(maxlen=64)
    recent_hit_hist: Deque[PageId] = collections.deque(maxlen=64)

    for decision_index, req in enumerate(requests):
        pid = req.page_id
        hit = pid in order
        if hit:
            order.move_to_end(pid)
            recent_req_hist.append(pid)
            recent_hit_hist.append(pid)
            continue
        if len(order) < capacity:
            order[pid] = None
            recent_req_hist.append(pid)
            continue

        candidates = list(order.keys())
        if score_start <= decision_index < score_end:
            decision_id = f"{trace_name}|cap={capacity}|t={decision_index}"
            for candidate in candidates:
                next_idx = next_reuse_request_index(
                    occurrence_index,
                    candidate,
                    decision_index,
                )
                yield ReuseTailObservation(
                    family=family,
                    trace_name=trace_name,
                    capacity=int(capacity),
                    decision_index=int(decision_index),
                    decision_id=decision_id,
                    candidate_page_id=candidate,
                    next_reuse_request_index=next_idx,
                    t=reuse_delay_t(next_idx, decision_index),
                )

        lru_victim = candidates[0]
        order.pop(lru_victim)
        order[pid] = None
        recent_req_hist.append(pid)


def nearest_rank_quantile(sorted_values: Sequence[int], q: float) -> Optional[float]:
    """Nearest-rank empirical quantile for finite T values.

    Returns None for an empty finite sample.  This explicit definition avoids
    ambiguity over interpolation for integer request-count delays.
    """
    if not sorted_values:
        return None
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"q must be in [0,1], got {q}")
    rank = max(1, math.ceil(q * len(sorted_values)))
    return float(sorted_values[rank - 1])


def summarize_accumulator(acc: CellAccumulator, horizon: int) -> Dict[str, object]:
    h = int(horizon)
    observations = int(acc.observations)
    finite_count = int(acc.finite_reuse_count)
    never_count = int(acc.never_reused_count)
    sorted_t = sorted(acc.finite_t_values)

    def frac(numer: int, denom: int) -> Optional[float]:
        return (float(numer) / float(denom)) if denom else None

    row: Dict[str, object] = {
        "family": acc.family,
        "trace": acc.trace_name,
        "capacity": int(acc.capacity),
        "horizon": h,
        "decision_points": int(acc.decision_points),
        "resident_candidate_observations": observations,
        "finite_reuse_count": finite_count,
        "never_reused_count": never_count,
        "t_gt_h_count_including_never": int(acc.exceed_counts[h]),
        "t_gt_h_count_eventually_reused": int(acc.finite_exceed_counts[h]),
        "t_le_h_count": int(acc.within_counts[h]),
        "never_reused_fraction": frac(never_count, observations),
        "p_t_gt_h_including_never": frac(int(acc.exceed_counts[h]), observations),
        "p_t_gt_h_eventually_reused": frac(int(acc.finite_exceed_counts[h]), finite_count),
        "p_t_le_h_including_never": frac(int(acc.within_counts[h]), observations),
        "finite_t_min": float(sorted_t[0]) if sorted_t else None,
        "finite_t_max": float(sorted_t[-1]) if sorted_t else None,
    }
    for q in QUANTILE_LEVELS:
        row[f"finite_t_q{int(q * 100):02d}"] = nearest_rank_quantile(sorted_t, q)
    return row


def merge_summary_rows(rows: Sequence[Mapping[str, object]], *, group: Mapping[str, object]) -> Dict[str, object]:
    """Merge compatible summary rows for one horizon into an aggregate row."""
    if not rows:
        raise ValueError("cannot merge an empty row set")
    h = int(rows[0]["horizon"])
    observations = sum(int(r["resident_candidate_observations"]) for r in rows)
    finite_count = sum(int(r["finite_reuse_count"]) for r in rows)
    never_count = sum(int(r["never_reused_count"]) for r in rows)
    exceed = sum(int(r["t_gt_h_count_including_never"]) for r in rows)
    finite_exceed = sum(int(r["t_gt_h_count_eventually_reused"]) for r in rows)
    within = sum(int(r["t_le_h_count"]) for r in rows)

    def frac(numer: int, denom: int) -> Optional[float]:
        return (float(numer) / float(denom)) if denom else None

    out = {
        **group,
        "horizon": h,
        "decision_points": sum(int(r["decision_points"]) for r in rows),
        "resident_candidate_observations": observations,
        "finite_reuse_count": finite_count,
        "never_reused_count": never_count,
        "t_gt_h_count_including_never": exceed,
        "t_gt_h_count_eventually_reused": finite_exceed,
        "t_le_h_count": within,
        "never_reused_fraction": frac(never_count, observations),
        "p_t_gt_h_including_never": frac(exceed, observations),
        "p_t_gt_h_eventually_reused": frac(finite_exceed, finite_count),
        "p_t_le_h_including_never": frac(within, observations),
    }
    return out
