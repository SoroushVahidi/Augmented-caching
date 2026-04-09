"""Shared offline baseline runner helpers."""

from __future__ import annotations

from typing import Dict, Iterable, Protocol

from lafc.offline.types import OfflineSimulationResult
from lafc.types import Page, PageId, Request


class OfflineSolver(Protocol):
    """Protocol for offline baselines that consume full traces."""

    name: str

    def solve(
        self,
        requests: Iterable[Request],
        pages: Dict[PageId, Page],
        capacity: int,
        **kwargs: object,
    ) -> OfflineSimulationResult: ...


def run_offline_solver(
    solver: OfflineSolver,
    requests: Iterable[Request],
    pages: Dict[PageId, Page],
    capacity: int,
    **kwargs: object,
) -> OfflineSimulationResult:
    """Run any offline solver through one extension point."""
    return solver.solve(requests=requests, pages=pages, capacity=capacity, **kwargs)
