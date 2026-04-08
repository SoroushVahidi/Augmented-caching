"""
Prediction error metrics.

Three error measures are implemented, covering both papers:

**Bansal et al. SODA 2022 (Baseline 1 — weighted paging)**

1.  **η (eta)** — weighted absolute error:

        η = Σ_t  w_{σ_t} · |τ_t − a_t|

    where τ_t is the predicted next arrival and a_t is the actual next
    arrival of page σ_t at time t.

2.  **Weighted surprises** — per-weight-class inversion count.

    INTERPRETATION NOTE: The paper's "ε-like" weighted surprise metric
    is described informally in terms of inversions within weight classes.
    Our implementation counts, for each request t in weight class W_i,
    whether the prediction τ_t disagrees with the actual a_t (τ_t ≠ a_t).
    The per-class weighted surprise is:
        weighted_surprise_i = w_i × (number of disagreements in class i)
    Total weighted surprise = Σ_i weighted_surprise_i.

**Lykouris & Vassilvitskii ICML 2018 / JACM 2021 (Baseline 2 — unweighted paging)**

3.  **η_unweighted** — total absolute error:

        η = Σ_t  |τ_t − a_t|

    All pages have unit cost, so weights drop out.

**Antoniadis et al. ICML 2020 (Baseline 3 — TRUST&DOUBT)**

4.  **η_discrete** — discrete (count-based) prediction error:

        η_discrete = |{t : τ_t ≠ a_t}|

    Counts the number of requests where the predicted next arrival
    differs from the actual next arrival.  Both τ_t = a_t = ∞ counts as
    agreement (zero error).  One-sided ∞ counts as one error.

    INTERPRETATION NOTE: The ICML 2020 paper characterises the error
    measure for the general MTS setting; the discrete count is the
    natural specialisation for the unweighted caching case where
    predictions are next-arrival times and the cost structure is binary.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from lafc.types import Page, PageId, Request


def compute_eta(
    requests: List[Request],
    pages: Dict[PageId, Page],
) -> float:
    """Compute the weighted prediction error η (Bansal et al. SODA 2022).

        η = Σ_t  w_{σ_t} · |τ_t − a_t|

    Parameters
    ----------
    requests:
        Trace with ``predicted_next`` and ``actual_next`` filled in.
    pages:
        Page weights.

    Returns
    -------
    float
        Total weighted prediction error.  Returns ``math.inf`` if any
        request has one-sided infinity in (τ_t, a_t).
    """
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


def compute_weighted_surprises(
    requests: List[Request],
    pages: Dict[PageId, Page],
) -> Dict[str, Any]:
    """Compute per-weight-class surprises and total weighted surprise.

    For each weight class (group of pages with equal weight w):
    - Count the number of requests t in the class where τ_t ≠ a_t.
    - weighted_surprise = w × count.

    INTERPRETATION NOTE:
        The paper defines a finer notion of "weighted surprises" based on
        inversions in the ordering induced by τ vs. a within each class.
        The count-of-disagreements metric here is an upper bound on that
        quantity.  See module docstring.

    Returns
    -------
    dict with keys:
        ``"per_class"``
            Dict mapping weight (as str) to
            ``{"surprises": int, "weighted_surprise": float}``.
        ``"total_surprises"``
            Sum of surprise counts across all classes.
        ``"total_weighted_surprise"``
            Sum of weighted surprises across all classes.
    """
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

    return {
        "per_class": per_class,
        "total_surprises": total_surprises,
        "total_weighted_surprise": total_weighted,
    }


def compute_eta_unweighted(requests: List[Request]) -> float:
    """Compute the unweighted prediction error η (Lykouris & Vassilvitskii 2018).

        η = Σ_t  |τ_t − a_t|

    All pages have unit cost in the standard (unweighted) paging setting,
    so weights do not appear.

    Parameters
    ----------
    requests:
        Trace with ``predicted_next`` and ``actual_next`` filled in.

    Returns
    -------
    float
        Total prediction error.  Returns ``math.inf`` if any request has
        one-sided infinity in (τ_t, a_t).
    """
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


def compute_discrete_eta(requests: List[Request]) -> int:
    """Compute the discrete prediction error η (Antoniadis et al. ICML 2020).

        η_discrete = |{t : τ_t ≠ a_t}|

    Counts the number of requests where the predicted next arrival differs
    from the actual next arrival.  Both being ∞ counts as agreement (η = 0).
    One-sided ∞ counts as one disagreement.

    Parameters
    ----------
    requests:
        Trace with ``predicted_next`` and ``actual_next`` filled in.

    Returns
    -------
    int
        Number of prediction disagreements (0 ≤ η_discrete ≤ T).
    """
    count = 0
    for req in requests:
        tau = req.predicted_next
        a = req.actual_next
        if math.isinf(tau) and math.isinf(a):
            pass  # both agree "never again"
        elif math.isinf(tau) or math.isinf(a):
            count += 1  # one-sided ∞ is a disagreement
        elif tau != a:
            count += 1
    return count
