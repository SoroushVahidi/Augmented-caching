from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, median
from typing import Dict, Iterable, Mapping, Optional, Sequence

from lafc.types import PageId


@dataclass(frozen=True)
class TargetTieMetrics:
    best_value: float
    optimal_candidates: tuple[PageId, ...]
    optimal_set_size: int
    optimal_set_fraction: float
    distinct_value_count: int
    ordinary_margin: float
    strict_distinct_margin: Optional[float]
    target_entropy_bits: float
    target_spread: float
    candidate_count: int


@dataclass(frozen=True)
class TieResolutionMetrics:
    h_long: int
    tie_set_size: int
    long_min: float
    long_max: float
    long_spread: float
    tied_set_remains_tied: bool
    tied_set_broken: bool
    deterministic_choice: PageId
    deterministic_long_value: float
    deterministic_is_long_best: bool
    deterministic_long_regret: float
    learned_choice: Optional[PageId] = None
    learned_long_value: Optional[float] = None
    learned_is_long_best: Optional[bool] = None
    learned_long_regret: Optional[float] = None


def eviction_loss_values(candidate_rows: Sequence[Mapping[str, object]]) -> Dict[PageId, float]:
    return {str(row["candidate_page_id"]): float(row["eviction_loss_label"]) for row in candidate_rows}


def distinct_sorted_values(values: Mapping[PageId, float]) -> list[float]:
    return sorted(set(float(value) for value in values.values()))


def exact_tie_metrics(values: Mapping[PageId, float]) -> TargetTieMetrics:
    if not values:
        raise ValueError("values must not be empty")
    candidate_count = len(values)
    sorted_values = sorted(float(value) for value in values.values())
    distinct = distinct_sorted_values(values)
    best = distinct[0]
    optimal = tuple(sorted(pid for pid, value in values.items() if float(value) == best))
    counts: Dict[float, int] = {}
    for value in values.values():
        counts[float(value)] = counts.get(float(value), 0) + 1
    entropy = 0.0
    for count in counts.values():
        p = count / candidate_count
        entropy -= p * math.log2(p)
    return TargetTieMetrics(
        best_value=best,
        optimal_candidates=optimal,
        optimal_set_size=len(optimal),
        optimal_set_fraction=len(optimal) / candidate_count,
        distinct_value_count=len(distinct),
        ordinary_margin=(sorted_values[1] - sorted_values[0]) if candidate_count > 1 else 0.0,
        strict_distinct_margin=(distinct[1] - distinct[0]) if len(distinct) > 1 else None,
        target_entropy_bits=entropy,
        target_spread=distinct[-1] - distinct[0],
        candidate_count=candidate_count,
    )


def deterministic_exact_tiebreak(optimal_candidates: Sequence[PageId]) -> PageId:
    if not optimal_candidates:
        raise ValueError("optimal_candidates must not be empty")
    return sorted(str(candidate) for candidate in optimal_candidates)[0]


def resolve_tied_set_at_long_horizon(
    *,
    h_long: int,
    h_tied_candidates: Sequence[PageId],
    long_values: Mapping[PageId, float],
    deterministic_choice: PageId,
    learned_choice: Optional[PageId] = None,
) -> TieResolutionMetrics:
    tied = tuple(str(candidate) for candidate in h_tied_candidates)
    if not tied:
        raise ValueError("h_tied_candidates must not be empty")
    missing = sorted(candidate for candidate in tied if candidate not in long_values)
    if missing:
        raise ValueError(f"long_values missing tied candidates: {missing}")
    if deterministic_choice not in tied:
        raise ValueError(f"deterministic choice {deterministic_choice!r} is not in the H-tied set")
    values = {candidate: float(long_values[candidate]) for candidate in tied}
    long_min = min(values.values())
    long_max = max(values.values())
    long_best = {candidate for candidate, value in values.items() if value == long_min}
    learned_long_value: Optional[float] = None
    learned_is_best: Optional[bool] = None
    learned_regret: Optional[float] = None
    if learned_choice is not None:
        if learned_choice not in tied:
            raise ValueError(f"learned choice {learned_choice!r} is not in the H-tied set")
        learned_long_value = values[learned_choice]
        learned_is_best = learned_choice in long_best
        learned_regret = learned_long_value - long_min
    deterministic_long_value = values[deterministic_choice]
    return TieResolutionMetrics(
        h_long=int(h_long),
        tie_set_size=len(tied),
        long_min=long_min,
        long_max=long_max,
        long_spread=long_max - long_min,
        tied_set_remains_tied=(long_max == long_min),
        tied_set_broken=(long_max != long_min),
        deterministic_choice=deterministic_choice,
        deterministic_long_value=deterministic_long_value,
        deterministic_is_long_best=deterministic_choice in long_best,
        deterministic_long_regret=deterministic_long_value - long_min,
        learned_choice=learned_choice,
        learned_long_value=learned_long_value,
        learned_is_long_best=learned_is_best,
        learned_long_regret=learned_regret,
    )


def numeric_summary(values: Iterable[float]) -> Dict[str, Optional[float]]:
    xs = sorted(float(value) for value in values)
    if not xs:
        return {"mean": None, "median": None, "p90": None, "min": None, "max": None}
    p90_idx = min(len(xs) - 1, math.ceil(0.9 * len(xs)) - 1)
    return {
        "mean": float(mean(xs)),
        "median": float(median(xs)),
        "p90": float(xs[p90_idx]),
        "min": float(xs[0]),
        "max": float(xs[-1]),
    }


__all__ = [
    "TargetTieMetrics",
    "TieResolutionMetrics",
    "deterministic_exact_tiebreak",
    "distinct_sorted_values",
    "eviction_loss_values",
    "exact_tie_metrics",
    "numeric_summary",
    "resolve_tied_set_at_long_horizon",
]
