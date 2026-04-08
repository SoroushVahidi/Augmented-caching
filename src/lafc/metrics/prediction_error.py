"""Prediction-error metrics for paging baselines."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Set

from lafc.types import Page, PageId, Request


def compute_eta(requests: List[Request], pages: Dict[PageId, Page]) -> float:
    eta = 0.0
    for req in requests:
        tau = req.predicted_next
        a = req.actual_next
        w = pages[req.page_id].weight
        if math.isinf(tau) and math.isinf(a):
            diff = 0.0
        elif math.isinf(tau) or math.isinf(a):
            return math.inf
        else:
            diff = abs(tau - a)
        eta += w * diff
    return eta


def compute_weighted_surprises(requests: List[Request], pages: Dict[PageId, Page]) -> Dict[str, Any]:
    class_surprises: Dict[float, int] = {}
    for req in requests:
        w = pages[req.page_id].weight
        tau = req.predicted_next
        a = req.actual_next
        if math.isinf(tau) and math.isinf(a):
            is_surprise = False
        elif math.isinf(tau) or math.isinf(a):
            is_surprise = True
        else:
            is_surprise = tau != a
        if is_surprise:
            class_surprises[w] = class_surprises.get(w, 0) + 1

    per_class: Dict[str, Any] = {}
    total_surprises = 0
    total_weighted = 0.0
    for w, count in sorted(class_surprises.items()):
        ws = w * count
        per_class[str(w)] = {"surprises": count, "weighted_surprise": ws}
        total_surprises += count
        total_weighted += ws
    return {"per_class": per_class, "total_surprises": total_surprises, "total_weighted_surprise": total_weighted}


def compute_eta_unweighted(requests: List[Request]) -> float:
    eta = 0.0
    for req in requests:
        tau = req.predicted_next
        a = req.actual_next
        if math.isinf(tau) and math.isinf(a):
            diff = 0.0
        elif math.isinf(tau) or math.isinf(a):
            return math.inf
        else:
            diff = abs(tau - a)
        eta += diff
    return eta


def _belady_states(requests: List[Request], capacity: int) -> List[Set[PageId]]:
    cache: Set[PageId] = set()
    actual_next_by_page: Dict[PageId, float] = {}
    states: List[Set[PageId]] = []
    for req in requests:
        actual_next_by_page[req.page_id] = req.actual_next
        if req.page_id not in cache:
            if len(cache) == capacity:
                victim = max(cache, key=lambda q: (actual_next_by_page.get(q, math.inf), q))
                cache.remove(victim)
            cache.add(req.page_id)
        states.append(set(cache))
    return states


def compute_cache_state_error(requests: List[Request], capacity: int) -> Dict[str, Any]:
    """MTS-style prediction error for caching states.

    INTERPRETATION NOTE: for equal-size cache states, dist(X,Y) is taken as
    |X \ Y| (= |Y \ X|), i.e. number of replacements needed.
    """
    pred_states: List[Set[PageId]] = []
    for req in requests:
        pc = req.metadata.get("predicted_cache")
        if pc is None:
            return {"total_error": None, "per_step": []}
        pred_states.append(set(str(x) for x in pc))

    off_states = _belady_states(requests, capacity)
    per_step: List[Dict[str, Any]] = []
    total = 0
    for t, (p, o) in enumerate(zip(pred_states, off_states)):
        d = len(p - o)
        total += d
        per_step.append({"t": t, "error": d})
    return {"total_error": total, "per_step": per_step}
