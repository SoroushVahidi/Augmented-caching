"""
Prediction corruption utilities for robustness experiments.

These helpers produce *noisy* variants of a perfect or given prediction
sequence, allowing experiments that measure how learning-augmented
algorithms degrade as prediction quality decreases.

All randomness is seeded via :class:`random.Random` for reproducibility.
No external dependencies (stdlib only).
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

from lafc.types import Page, PageId, Request


def add_additive_noise(
    requests: List[Request],
    sigma: float,
    rng: Optional[random.Random] = None,
) -> List[Request]:
    """Add Gaussian-like noise to each ``predicted_next``.

    Noise is sampled from a Gaussian with mean 0 and standard deviation
    *sigma* using the Box-Muller transform (stdlib only).  Results are
    clipped to ``[t + 1, math.inf)`` so that predictions remain after the
    current time step.

    Parameters
    ----------
    requests:
        Input request list (not modified in place).
    sigma:
        Standard deviation of the additive noise.
    rng:
        Optional seeded :class:`random.Random` instance.  A new unseeded
        instance is created if ``None``.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    _rng = rng or random.Random()
    result: List[Request] = []
    for req in requests:
        tau = req.predicted_next
        if math.isinf(tau):
            noisy_tau = tau  # don't corrupt "never requested again" sentinel
        else:
            noise = _rng.gauss(0.0, sigma)
            noisy_tau = max(float(req.t + 1), tau + noise)
        result.append(
            Request(
                t=req.t,
                page_id=req.page_id,
                predicted_next=noisy_tau,
                actual_next=req.actual_next,
            )
        )
    return result


def random_swap_within_class(
    requests: List[Request],
    pages: Dict[PageId, Page],
    swap_prob: float,
    rng: Optional[random.Random] = None,
) -> List[Request]:
    """Randomly swap predicted-next values between requests in the same weight class.

    With probability *swap_prob* each request's ``predicted_next`` is
    exchanged with a randomly chosen other request whose page belongs to the
    same weight class.  This introduces local inversions without changing the
    global distribution of predictions.

    Parameters
    ----------
    requests:
        Input request list (not modified in place).
    pages:
        Page weight dictionary (used to determine weight classes).
    swap_prob:
        Probability in ``[0, 1]`` that any given request participates in a swap.
    rng:
        Optional seeded RNG.
    """
    if not (0.0 <= swap_prob <= 1.0):
        raise ValueError(f"swap_prob must be in [0,1], got {swap_prob}")

    _rng = rng or random.Random()

    # Group request indices by weight class.
    from collections import defaultdict
    class_indices: Dict[float, List[int]] = defaultdict(list)
    for req in requests:
        w = pages[req.page_id].weight
        class_indices[w].append(req.t)

    preds = [req.predicted_next for req in requests]

    for w, indices in class_indices.items():
        if len(indices) < 2:
            continue
        for i in indices:
            if _rng.random() < swap_prob:
                j = _rng.choice(indices)
                preds[i], preds[j] = preds[j], preds[i]

    return [
        Request(
            t=req.t,
            page_id=req.page_id,
            predicted_next=preds[req.t],
            actual_next=req.actual_next,
        )
        for req in requests
    ]


def bounded_inversions(
    requests: List[Request],
    max_inversions: int,
    rng: Optional[random.Random] = None,
) -> List[Request]:
    """Introduce at most *max_inversions* prediction inversions.

    An inversion is created by swapping the ``predicted_next`` values of a
    randomly chosen pair of requests (any pair, regardless of weight class).

    Parameters
    ----------
    requests:
        Input request list (not modified in place).
    max_inversions:
        Maximum number of pair-swaps to perform.
    rng:
        Optional seeded RNG.
    """
    if max_inversions < 0:
        raise ValueError(f"max_inversions must be >= 0, got {max_inversions}")

    _rng = rng or random.Random()
    n = len(requests)
    preds = [req.predicted_next for req in requests]

    num_swaps = min(max_inversions, n * (n - 1) // 2)
    indices = list(range(n))

    for _ in range(num_swaps):
        i, j = _rng.sample(indices, 2)
        preds[i], preds[j] = preds[j], preds[i]

    return [
        Request(
            t=req.t,
            page_id=req.page_id,
            predicted_next=preds[req.t],
            actual_next=req.actual_next,
        )
        for req in requests
    ]
