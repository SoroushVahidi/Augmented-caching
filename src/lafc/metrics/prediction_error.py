"""
Prediction error metrics from the paper.

Reference
---------
Bansal, Coester, Kumar, Purohit, Vee.
"Learning-Augmented Weighted Paging."
SODA 2022.

Two error measures are implemented:

1.  **η (eta)** — weighted absolute error (Definition in the paper):

        η = Σ_t  w_{σ_t} · |τ_t − a_t|

    where τ_t is the predicted next arrival and a_t is the actual next
    arrival of page σ_t at time t.

    Handling of infinities:
    - Both τ_t = a_t = ∞ → contribution is 0 (both agree "never again").
    - Only one is ∞         → contribution is ∞ (maximum disagreement).

2.  **Weighted surprises** — per-weight-class inversion count.

    INTERPRETATION NOTE: The paper's "ε-like" weighted surprise metric
    is described informally in terms of inversions within weight classes.
    Our implementation counts, for each request t in weight class W_i,
    whether the prediction τ_t disagrees with the actual a_t (τ_t ≠ a_t).
    The per-class weighted surprise is:
        weighted_surprise_i = w_i × (number of disagreements in class i)
    Total weighted surprise = Σ_i weighted_surprise_i.

    A stricter inversion-based count (tracking permutation inversions in
    the τ ordering vs. the a ordering within each class) would be closer
    to the paper's formal definition but requires O(T²) comparisons per
    class and is beyond the scope of this baseline.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from lafc.types import Page, PageId, Request


def compute_eta(
    requests: List[Request],
    pages: Dict[PageId, Page],
) -> float:
    """Compute the weighted prediction error η.

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
            # Both agree "never requested again" — no error.
            diff = 0.0
        elif math.isinf(tau) or math.isinf(a):
            # One side says "never", the other gives a finite time — max error.
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
        quantity: every inversion pair is a disagreement, but not every
        disagreement forms an inversion pair.  See module docstring.

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
    # Accumulate per weight-class counts.
    class_surprises: Dict[float, int] = {}
    for req in requests:
        w = pages[req.page_id].weight
        tau = req.predicted_next
        a = req.actual_next

        # Determine whether prediction differs from actual.
        if math.isinf(tau) and math.isinf(a):
            is_surprise = False
        elif math.isinf(tau) or math.isinf(a):
            is_surprise = True
        else:
            is_surprise = tau != a

        if is_surprise:
            class_surprises[w] = class_surprises.get(w, 0) + 1

    # Build output structure.
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
