"""Core shared dataclasses for LAFC."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

PageId = str


@dataclass
class Page:
    """A page with its fetch cost (weight)."""

    page_id: PageId
    weight: float

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError(f"Page '{self.page_id}': weight must be > 0, got {self.weight}")


@dataclass
class Request:
    """One request in a trace."""

    t: int
    page_id: PageId
    predicted_next: float = math.inf
    actual_next: float = math.inf
    # Optional extensible payload (e.g. predicted cache configuration P_t).
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheEvent:
    """Policy decision for one request."""

    t: int
    page_id: PageId
    hit: bool
    cost: float
    evicted: Optional[PageId] = None
    phase: Optional[int] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Aggregated simulation output."""

    policy_name: str
    total_cost: float
    total_hits: int
    total_misses: int
    events: List[CacheEvent] = field(default_factory=list)
    prediction_error_eta: Optional[float] = None
    prediction_error_surprises: Optional[Dict[str, Any]] = None
    extra_diagnostics: Optional[Dict[str, Any]] = None
