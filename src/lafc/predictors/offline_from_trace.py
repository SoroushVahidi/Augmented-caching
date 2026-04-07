"""
Offline (perfect) prediction generator.

For each request at time t for page p, the perfect prediction is
τ_t = actual next arrival of p after t, or math.inf if p never appears again.

This is useful as an oracle baseline and for testing the consistency
property of learning-augmented algorithms.
"""

from __future__ import annotations

import math
from typing import Dict, List

from lafc.types import PageId, Request


def extract_actual_next_arrivals(page_ids: List[PageId]) -> List[float]:
    """For each index t, return the next index t' > t where page_ids[t'] == page_ids[t].

    Returns math.inf if the page never appears again.

    This is a pure utility over raw page-id lists; it does not require
    Request objects to already exist.
    """
    n = len(page_ids)
    result: List[float] = [math.inf] * n
    last_seen: Dict[PageId, int] = {}
    for t in range(n - 1, -1, -1):
        pid = page_ids[t]
        if pid in last_seen:
            result[t] = float(last_seen[pid])
        last_seen[pid] = t
    return result


def compute_perfect_predictions(requests: List[Request]) -> List[Request]:
    """Return a new list of Request objects with ``predicted_next = actual_next``.

    This simulates a clairvoyant oracle that predicts each page's next
    arrival perfectly.  The ``actual_next`` values in the input are
    assumed to already be filled in (e.g. by
    :func:`~lafc.simulator.request_trace.build_requests_from_lists`).
    """
    return [
        Request(
            t=req.t,
            page_id=req.page_id,
            predicted_next=req.actual_next,  # perfect prediction
            actual_next=req.actual_next,
        )
        for req in requests
    ]
