"""
Request trace loader.

Supports a simple JSON format:

.. code-block:: json

    {
        "requests":    ["A", "B", "C", "A"],
        "weights":     {"A": 1.0, "B": 2.0, "C": 4.0},
        "predictions": [3, 5, 9999, 9999]
    }

``predictions`` is optional.  When absent every ``predicted_next`` is set to
``math.inf`` (treat all pages as "never needed again").

``actual_next`` is always computed from the trace itself and is not read from
the file, because it is derived ground-truth rather than an input.
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Tuple

from lafc.types import Page, PageId, Request


def _compute_actual_next(page_ids: List[PageId]) -> List[float]:
    """For each index t, return the smallest t' > t with page_ids[t'] == page_ids[t].

    Returns ``math.inf`` if the page never appears again.
    """
    n = len(page_ids)
    result: List[float] = [math.inf] * n
    # Walk backwards; keep the most recent occurrence of each page id.
    last_seen: Dict[PageId, int] = {}
    for t in range(n - 1, -1, -1):
        pid = page_ids[t]
        if pid in last_seen:
            result[t] = float(last_seen[pid])
        last_seen[pid] = t
    return result


def build_requests_from_lists(
    page_ids: List[PageId],
    weights: Dict[PageId, float],
    predictions: Optional[List[float]] = None,
) -> Tuple[List[Request], Dict[PageId, Page]]:
    """Build a request list and page dictionary from raw lists.

    Parameters
    ----------
    page_ids:
        Ordered list of requested page identifiers.
    weights:
        Mapping from page identifier to fetch cost.  All page ids that appear
        in *page_ids* must have an entry here.
    predictions:
        Optional list of predicted next-arrival times aligned with *page_ids*.
        Length must equal ``len(page_ids)`` when provided.

    Returns
    -------
    requests:
        List of :class:`~lafc.types.Request` objects with ``actual_next``
        and ``predicted_next`` filled in.
    pages:
        Dictionary of :class:`~lafc.types.Page` objects for every unique page
        referenced in the trace.
    """
    if not page_ids:
        raise ValueError("page_ids must not be empty")

    # Validate that all requested pages have weights.
    missing = [pid for pid in page_ids if pid not in weights]
    if missing:
        raise KeyError(f"No weight provided for page(s): {sorted(set(missing))}")

    # Validate weights > 0.
    for pid, w in weights.items():
        if w <= 0:
            raise ValueError(f"Weight for page '{pid}' must be > 0, got {w}")

    if predictions is not None and len(predictions) != len(page_ids):
        raise ValueError(
            f"len(predictions)={len(predictions)} != len(page_ids)={len(page_ids)}"
        )

    actual_nexts = _compute_actual_next(page_ids)
    preds = predictions if predictions is not None else [math.inf] * len(page_ids)

    requests: List[Request] = [
        Request(
            t=t,
            page_id=pid,
            predicted_next=float(preds[t]),
            actual_next=actual_nexts[t],
        )
        for t, pid in enumerate(page_ids)
    ]

    # Build Page objects for every unique page id in the trace.
    pages: Dict[PageId, Page] = {
        pid: Page(page_id=pid, weight=weights[pid])
        for pid in set(page_ids)
    }

    return requests, pages


def load_trace(path: str) -> Tuple[List[Request], Dict[PageId, Page]]:
    """Load a weighted paging trace from a JSON file.

    See module docstring for the expected file format.

    Parameters
    ----------
    path:
        Filesystem path to a JSON trace file.

    Returns
    -------
    Same as :func:`build_requests_from_lists`.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if "requests" not in data:
        raise ValueError(f"Trace file '{path}' is missing 'requests' field")
    if "weights" not in data:
        raise ValueError(f"Trace file '{path}' is missing 'weights' field")

    page_ids: List[PageId] = [str(p) for p in data["requests"]]
    weights: Dict[PageId, float] = {str(k): float(v) for k, v in data["weights"].items()}
    predictions: Optional[List[float]] = (
        [float(x) for x in data["predictions"]]
        if "predictions" in data
        else None
    )

    return build_requests_from_lists(page_ids, weights, predictions)
