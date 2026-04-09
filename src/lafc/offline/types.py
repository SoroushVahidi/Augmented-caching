"""Data structures for offline caching baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from lafc.types import PageId


@dataclass(frozen=True)
class OfflineDecision:
    """Per-request decision emitted by an offline solver."""

    t: int
    page_id: PageId
    hit: bool
    cost: float
    evicted: Optional[PageId] = None
    evicted_next_use: Optional[int] = None
    evicted_next_use_distance: Optional[int] = None
    evicted_never_used_again: bool = False
    tie_size: int = 1
    inserted: bool = False
    bypassed: bool = False
    cache_occupancy: Optional[float] = None


@dataclass
class OfflineSimulationResult:
    """Aggregated output for an offline baseline run."""

    solver_name: str
    capacity: Union[int, float]
    total_requests: int
    total_hits: int
    total_misses: int
    decisions: List[OfflineDecision]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        return 0.0 if self.total_requests == 0 else self.total_hits / self.total_requests

    @property
    def total_cost(self) -> float:
        return sum(float(d.cost) for d in self.decisions)
