"""
Tests for cost and prediction-error metrics.
"""

from __future__ import annotations

import math

import pytest

from lafc.metrics.cost import hit_rate, per_page_cost, total_fetch_cost, total_hits, total_misses
from lafc.metrics.prediction_error import compute_eta, compute_weighted_surprises
from lafc.simulator.request_trace import build_requests_from_lists
from lafc.types import CacheEvent, Page


# ---------------------------------------------------------------------------
# Cost metrics
# ---------------------------------------------------------------------------


def _events(*args):
    """Build a list of CacheEvents from (hit, cost, page_id) tuples."""
    return [
        CacheEvent(t=i, page_id=pid, hit=hit, cost=cost)
        for i, (hit, cost, pid) in enumerate(args)
    ]


def test_total_fetch_cost():
    events = _events((True, 0.0, "A"), (False, 2.0, "B"), (False, 3.0, "C"))
    assert total_fetch_cost(events) == pytest.approx(5.0)


def test_total_hits_and_misses():
    events = _events((True, 0.0, "A"), (False, 1.0, "B"), (True, 0.0, "C"), (False, 2.0, "D"))
    assert total_hits(events) == 2
    assert total_misses(events) == 2


def test_hit_rate_basic():
    events = _events((True, 0.0, "A"), (True, 0.0, "B"), (False, 1.0, "C"), (False, 1.0, "D"))
    assert hit_rate(events) == pytest.approx(0.5)


def test_hit_rate_empty():
    assert hit_rate([]) == 0.0


def test_hit_rate_all_hits():
    events = _events((True, 0.0, "A"), (True, 0.0, "B"))
    assert hit_rate(events) == pytest.approx(1.0)


def test_per_page_cost():
    events = _events(
        (False, 1.0, "A"),
        (False, 2.0, "B"),
        (False, 1.0, "A"),
        (True, 0.0, "B"),
    )
    costs = per_page_cost(events)
    assert costs["A"] == pytest.approx(2.0)
    assert costs["B"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Prediction error metrics
# ---------------------------------------------------------------------------


def _make_pages(**kwargs):
    return {pid: Page(page_id=pid, weight=w) for pid, w in kwargs.items()}


def test_eta_perfect_predictions_is_zero():
    requests, pages = build_requests_from_lists(
        ["A", "B", "A"], {"A": 1.0, "B": 2.0}
    )
    from lafc.predictors.offline_from_trace import compute_perfect_predictions
    requests_perf = compute_perfect_predictions(requests)
    eta = compute_eta(requests_perf, pages)
    assert eta == pytest.approx(0.0)


def test_eta_imperfect_predictions():
    """τ_t = a_t + 2 for all t; η = Σ w_p * 2."""
    requests, pages = build_requests_from_lists(
        ["A", "B"],
        {"A": 1.0, "B": 3.0},
        predictions=[999.0, 999.0],
    )
    # actual_next: A@t=0 → inf, B@t=1 → inf; tau=999 → diff=inf for both... not ideal.
    # Use a trace where actual_next is finite.
    requests2, pages2 = build_requests_from_lists(
        ["A", "B", "A", "B"],
        {"A": 1.0, "B": 3.0},
        predictions=[4.0, 5.0, 9999.0, 9999.0],
    )
    # actual_next: A@t=0=2, B@t=1=3, A@t=2=inf, B@t=3=inf
    # Error at t=0: |4 - 2| * 1 = 2
    # Error at t=1: |5 - 3| * 3 = 6
    # t=2,t=3: tau=9999 vs actual=inf → inf... hmm, use actual_next-aligned preds.
    from lafc.predictors.offline_from_trace import compute_perfect_predictions
    requests_perf = compute_perfect_predictions(requests2)
    # Now add a fixed offset to all finite predictions.
    from lafc.types import Request
    noisy = [
        Request(
            t=r.t,
            page_id=r.page_id,
            predicted_next=r.actual_next + 2 if not math.isinf(r.actual_next) else r.actual_next,
            actual_next=r.actual_next,
        )
        for r in requests_perf
    ]
    eta = compute_eta(noisy, pages2)
    # t=0: w=1 * |2+2 - 2| = 2; t=1: w=3 * |3+2 - 3| = 6; t=2,3: both inf → 0
    assert eta == pytest.approx(2.0 + 6.0)


def test_eta_one_sided_inf():
    """One-sided infinity should yield math.inf."""
    from lafc.types import Request
    pages = _make_pages(A=1.0)
    req = [Request(t=0, page_id="A", predicted_next=math.inf, actual_next=5.0)]
    eta = compute_eta(req, pages)
    assert math.isinf(eta)


def test_eta_both_inf_is_zero():
    """Both tau and a_t = inf means perfect agreement on "never again"."""
    from lafc.types import Request
    pages = _make_pages(A=2.0)
    req = [Request(t=0, page_id="A", predicted_next=math.inf, actual_next=math.inf)]
    eta = compute_eta(req, pages)
    assert eta == pytest.approx(0.0)


def test_weighted_surprises_structure():
    """Return value must have the required keys."""
    requests, pages = build_requests_from_lists(
        ["A", "B"], {"A": 1.0, "B": 2.0}
    )
    result = compute_weighted_surprises(requests, pages)
    assert "per_class" in result
    assert "total_surprises" in result
    assert "total_weighted_surprise" in result


def test_weighted_surprises_perfect_predictions_zero():
    """With perfect predictions there are no surprises."""
    requests, pages = build_requests_from_lists(
        ["A", "B", "A"], {"A": 1.0, "B": 2.0}
    )
    from lafc.predictors.offline_from_trace import compute_perfect_predictions
    requests_perf = compute_perfect_predictions(requests)
    result = compute_weighted_surprises(requests_perf, pages)
    assert result["total_surprises"] == 0
    assert result["total_weighted_surprise"] == pytest.approx(0.0)


def test_weighted_surprises_count():
    """Each request where τ ≠ a should be counted as a surprise."""
    # Use a trace with predictions that differ from actual.
    requests, pages = build_requests_from_lists(
        ["A", "B", "A", "B"],
        {"A": 1.0, "B": 2.0},
        predictions=[99.0, 99.0, 99.0, 99.0],
    )
    # actual_next: A@0=2, B@1=3, A@2=inf, B@3=inf
    # All four predictions differ from actual → 4 surprises.
    result = compute_weighted_surprises(requests, pages)
    assert result["total_surprises"] == 4
