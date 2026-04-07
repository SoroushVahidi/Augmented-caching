"""
Tests for all caching policies.

Hand-constructed traces allow exact cost verification.
"""

from __future__ import annotations

import math

import pytest

from lafc.policies.advice_trusting import AdviceTrustingPolicy
from lafc.policies.la_weighted_paging_deterministic import (
    LAWeightedPagingDeterministic,
    round_weight_to_power_of_2,
)
from lafc.policies.la_weighted_paging_randomized import LAWeightedPagingRandomized
from lafc.policies.lru import LRUPolicy
from lafc.policies.weighted_lru import WeightedLRUPolicy
from lafc.predictors.offline_from_trace import compute_perfect_predictions
from lafc.simulator.request_trace import build_requests_from_lists
from lafc.types import Page


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pages(*args, **weights):
    """Build a pages dict.  Call as _pages(A=1.0, B=2.0) or _pages(**{'A':1.0})."""
    return {pid: Page(page_id=pid, weight=w) for pid, w in weights.items()}


def _build(page_ids, weights, predictions=None):
    return build_requests_from_lists(page_ids, weights, predictions)


# ---------------------------------------------------------------------------
# LRU
# ---------------------------------------------------------------------------


def test_lru_basic_cost():
    """
    Trace: A B C A B C, capacity=2
    Expected:
      t=0 A miss cost=1  cache={A}
      t=1 B miss cost=1  cache={A,B}
      t=2 C miss cost=1  evict LRU=A  cache={B,C}
      t=3 A miss cost=1  evict LRU=B  cache={C,A}
      t=4 B miss cost=1  evict LRU=C  cache={A,B}
      t=5 C miss cost=1  evict LRU=A  cache={B,C}
    Total: 6 misses, cost=6
    """
    weights = {"A": 1.0, "B": 1.0, "C": 1.0}
    requests, pages = _build(["A", "B", "C", "A", "B", "C"], weights)
    policy = LRUPolicy()
    policy.reset(2, pages)
    events = [policy.on_request(r) for r in requests]
    assert all(not e.hit for e in events)
    assert policy.total_cost() == pytest.approx(6.0)


def test_lru_hit():
    """t=0 A miss, t=1 A hit."""
    requests, pages = _build(["A", "A"], {"A": 5.0})
    policy = LRUPolicy()
    policy.reset(2, pages)
    e0 = policy.on_request(requests[0])
    e1 = policy.on_request(requests[1])
    assert not e0.hit
    assert e1.hit
    assert e0.cost == pytest.approx(5.0)
    assert e1.cost == pytest.approx(0.0)


def test_lru_evicts_least_recently_used():
    """Access A then B, then request A again — B should be evicted when C arrives."""
    requests, pages = _build(["A", "B", "A", "C"], {"A": 1.0, "B": 1.0, "C": 1.0})
    policy = LRUPolicy()
    policy.reset(2, pages)
    events = [policy.on_request(r) for r in requests]
    # t=2: A hit (A is most recent), t=3: C miss, should evict B (LRU)
    assert events[2].hit
    assert events[3].evicted == "B"


# ---------------------------------------------------------------------------
# WeightedLRU
# ---------------------------------------------------------------------------


def test_weighted_lru_evicts_cheapest():
    """Cache has A(w=1) and B(w=4); on a miss should evict A."""
    requests, pages = _build(
        ["A", "B", "C"],
        {"A": 1.0, "B": 4.0, "C": 2.0},
    )
    policy = WeightedLRUPolicy()
    policy.reset(2, pages)
    events = [policy.on_request(r) for r in requests]
    # After t=0,1: cache={A,B}.  t=2: C miss, evict A (cheaper).
    assert events[2].evicted == "A"


def test_weighted_lru_tie_broken_by_lru():
    """When two cached pages have equal weight, evict the LRU one."""
    requests, pages = _build(
        ["A", "B", "A", "C"],
        {"A": 1.0, "B": 1.0, "C": 1.0},
    )
    policy = WeightedLRUPolicy()
    policy.reset(2, pages)
    events = [policy.on_request(r) for r in requests]
    # After t=0,1: A then B; t=2: A hit (A becomes MRU); cache={B,A}.
    # t=3: C miss; equal weight; evict LRU=B.
    assert events[3].evicted == "B"


# ---------------------------------------------------------------------------
# AdviceTrusting
# ---------------------------------------------------------------------------


def test_advice_trusting_evicts_farthest_predicted():
    """Cache has A(τ=100) and B(τ=5); on a miss should evict A."""
    weights = {"A": 1.0, "B": 1.0, "C": 1.0}
    requests, pages = _build(["A", "B", "C"], weights, predictions=[100.0, 5.0, 999.0])
    policy = AdviceTrustingPolicy()
    policy.reset(2, pages)
    events = [policy.on_request(r) for r in requests]
    assert events[2].evicted == "A"


def test_advice_trusting_perfect_predictions_optimal_simple():
    """With perfect predictions, advice_trusting acts like Belady.

    Trace: A B C A B C, capacity=2, all unit weights.
    Perfect predictions:
      τ(A@t=0)=3, τ(B@t=1)=4, τ(C@t=2)=5
      τ(A@t=3)=∞, τ(B@t=4)=∞, τ(C@t=5)=∞

    Execution:
      t=0: A miss, cache={A}
      t=1: B miss, cache={A,B}; τ[A]=3, τ[B]=4
      t=2: C miss; evict B (τ=4 > τ[A]=3); cache={A,C}; τ[C]=5
      t=3: A hit; τ[A]=∞
      t=4: B miss; evict A (τ=∞ > τ[C]=5); cache={C,B}; τ[B]=∞
      t=5: C hit; τ[C]=∞
    Total: 4 misses, cost=4.
    """
    weights = {"A": 1.0, "B": 1.0, "C": 1.0}
    requests, pages = _build(["A", "B", "C", "A", "B", "C"], weights)
    requests_perf = compute_perfect_predictions(requests)

    policy = AdviceTrustingPolicy()
    policy.reset(2, pages)
    for r in requests_perf:
        policy.on_request(r)
    assert policy.total_cost() == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# LA Deterministic
# ---------------------------------------------------------------------------


def test_la_det_runs_without_error():
    """The algorithm should process any valid trace without exceptions."""
    weights = {"A": 1.0, "B": 2.0, "C": 4.0}
    requests, pages = _build(["A", "B", "C", "A", "B", "C", "A"], weights)
    policy = LAWeightedPagingDeterministic()
    policy.reset(2, pages)
    for r in requests:
        policy.on_request(r)


def test_la_det_cost_accounting():
    """Total cost must equal sum of weights for each miss."""
    weights = {"A": 1.0, "B": 3.0, "C": 5.0}
    requests, pages = _build(["A", "B", "C", "A", "B", "C"], weights)
    policy = LAWeightedPagingDeterministic()
    policy.reset(2, pages)
    events = [policy.on_request(r) for r in requests]

    computed = sum(e.cost for e in events)
    assert computed == pytest.approx(policy.total_cost())
    # Each miss contributes the page's weight; verify no incorrect charging.
    for e in events:
        if e.hit:
            assert e.cost == 0.0
        else:
            assert e.cost == pytest.approx(weights[e.page_id])


def test_la_det_weight_classes_built_correctly():
    """Weight class structure should group pages by weight."""
    weights = {"A": 1.0, "B": 1.0, "C": 4.0}
    requests, pages = _build(["A", "B", "C"], weights)
    policy = LAWeightedPagingDeterministic()
    policy.reset(3, pages)
    wc = policy.weight_classes()
    assert 1.0 in wc
    assert 4.0 in wc
    assert wc[1.0].page_ids == {"A", "B"}
    assert wc[4.0].page_ids == {"C"}


def test_la_det_perfect_predictions_beats_lru_constructed():
    """
    Constructed trace where LA-DET with perfect predictions should do better
    than LRU.

    Trace: A B C B A, capacity=2, all unit weights.
    With perfect predictions:
      τ(A@t=0)=4, τ(B@t=1)=3, τ(C@t=2)=inf, τ(B@t=3)=inf, τ(A@t=4)=inf

    LRU:
      t=0: A miss, cache={A}
      t=1: B miss, cache={A,B}
      t=2: C miss, evict LRU=A, cache={B,C}
      t=3: B hit
      t=4: A miss, evict LRU=C, cache={B,A}
      Total: 4 misses, cost=4

    LA-DET with perfect predictions evicts the page predicted farthest away:
      t=0: A miss, cache={A}
      t=1: B miss, cache={A,B}
      t=2: C miss. Scores: A: τ=4/w=1=4, B: τ=3/w=1=3. Evict A. cache={B,C}
      t=3: B hit
      t=4: A miss. Evict C (τ=inf, highest score). cache={B,A}
      Total: 4 misses, cost=4

    Both get cost=4 here; LA-DET must at least match LRU.
    """
    weights = {"A": 1.0, "B": 1.0, "C": 1.0}
    page_ids = ["A", "B", "C", "B", "A"]
    requests, pages = _build(page_ids, weights)
    requests_perf = compute_perfect_predictions(requests)

    lru = LRUPolicy()
    lru.reset(2, pages)
    for r in requests:
        lru.on_request(r)

    la = LAWeightedPagingDeterministic()
    la.reset(2, pages)
    for r in requests_perf:
        la.on_request(r)

    # LA-DET should not be worse than LRU with perfect predictions.
    assert la.total_cost() <= lru.total_cost() + 1e-9


def test_la_det_prediction_update_on_hit():
    """On a cache hit, predicted_next should still be updated."""
    weights = {"A": 1.0}
    requests, pages = _build(["A", "A"], weights, predictions=[5.0, 9.0])
    policy = LAWeightedPagingDeterministic()
    policy.reset(2, pages)
    policy.on_request(requests[0])
    assert policy.predicted_next_snapshot()["A"] == 5.0
    policy.on_request(requests[1])
    assert policy.predicted_next_snapshot()["A"] == 9.0


def test_la_det_round_weights_option():
    """round_weights=True should round weights to powers of 2."""
    weights = {"A": 3.0, "B": 5.0}  # rounded to 4 and 8
    requests, pages = _build(["A", "B", "A"], weights)
    policy = LAWeightedPagingDeterministic(round_weights=True)
    policy.reset(3, pages)
    for r in requests:
        policy.on_request(r)
    wc = policy.weight_classes()
    assert 4.0 in wc  # 3 → 4
    assert 8.0 in wc  # 5 → 8


def test_la_det_deterministic_reproducibility():
    """Same trace + same predictions must always produce the same cost."""
    weights = {"A": 1.0, "B": 2.0, "C": 3.0}
    page_ids = ["A", "B", "C", "A", "B", "C", "A", "B", "C"]
    preds = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    requests, pages = _build(page_ids, weights, preds)

    costs = []
    for _ in range(3):
        policy = LAWeightedPagingDeterministic()
        policy.reset(2, pages)
        for r in requests:
            policy.on_request(r)
        costs.append(policy.total_cost())

    assert costs[0] == costs[1] == costs[2]


def test_la_det_eviction_score_logic():
    """
    With cache={A(w=1,τ=10), B(w=5,τ=10)}, requesting C:
    score(A) = 10/1 = 10
    score(B) = 10/5 = 2
    A should be evicted (higher score).
    """
    weights = {"A": 1.0, "B": 5.0, "C": 1.0}
    # A gets τ=10, B gets τ=10, then C is requested.
    requests, pages = _build(["A", "B", "C"], weights, predictions=[10.0, 10.0, 99.0])
    policy = LAWeightedPagingDeterministic()
    policy.reset(2, pages)
    events = [policy.on_request(r) for r in requests]
    assert events[2].evicted == "A"


# ---------------------------------------------------------------------------
# Randomized (scaffold check)
# ---------------------------------------------------------------------------


def test_randomized_raises_not_implemented():
    """The randomized policy must raise NotImplementedError, not silently fail."""
    policy = LAWeightedPagingRandomized()
    with pytest.raises(NotImplementedError):
        policy.reset(3, {})


# ---------------------------------------------------------------------------
# round_weight_to_power_of_2
# ---------------------------------------------------------------------------


def test_round_weight_exact_power():
    assert round_weight_to_power_of_2(4.0) == pytest.approx(4.0)
    assert round_weight_to_power_of_2(8.0) == pytest.approx(8.0)


def test_round_weight_non_power():
    assert round_weight_to_power_of_2(3.0) == pytest.approx(4.0)
    assert round_weight_to_power_of_2(5.0) == pytest.approx(8.0)
    assert round_weight_to_power_of_2(0.5) == pytest.approx(1.0)


def test_round_weight_zero_raises():
    with pytest.raises(ValueError):
        round_weight_to_power_of_2(0.0)
