"""
Core types for the Learning-Augmented Weighted Paging package.

All public dataclasses used across the package are defined here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

PageId = str  # Arbitrary hashable page identifier; str is the most general.


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------


@dataclass
class Page:
    """A page with its fetch cost (weight)."""

    page_id: PageId
    weight: float  # Must be > 0.

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError(
                f"Page '{self.page_id}': weight must be > 0, got {self.weight}"
            )


@dataclass
class Request:
    """A single timestep in a weighted paging trace.

    Attributes
    ----------
    t:
        Zero-based index of the request in the trace.
    page_id:
        The page being requested.
    predicted_next:
        Prediction for the next time index at which ``page_id`` will be
        requested again.  ``math.inf`` means "never again" (or unknown).
    actual_next:
        Ground-truth next occurrence of ``page_id`` after ``t``, derived
        from the full trace offline.  Available for evaluation only — the
        online algorithm must NOT use this.
    """

    t: int
    page_id: PageId
    predicted_next: float = math.inf
    actual_next: float = math.inf


@dataclass
class CacheEvent:
    """Records what happened at a single request step."""

    t: int
    page_id: PageId
    hit: bool
    cost: float  # 0.0 on a hit, w_p on a miss.
    evicted: Optional[PageId] = None  # Set when a page was evicted to make room.
    phase: Optional[int] = None  # Phase index (for phase-based policies such as Marker).


@dataclass
class SimulationResult:
    """Aggregated result of running a policy on a trace."""

    policy_name: str
    total_cost: float
    total_hits: int
    total_misses: int
    events: List[CacheEvent] = field(default_factory=list)
    # Filled in by the runner after the simulation.
    prediction_error_eta: Optional[float] = None
    prediction_error_surprises: Optional[Dict[str, Any]] = None
    # Optional extra diagnostics (e.g. phase/clean-chain info from Predictive Marker).
    extra_diagnostics: Optional[Dict[str, Any]] = None
