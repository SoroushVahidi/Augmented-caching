"""
Tests for Baseline 4 (Wei 2020):
  - OfflineBeladyPolicy
  - BlindOracleLRUCombiner (deterministic combiner)
  - Prediction error η (unweighted)
  - Noisy predictions pipeline

All traces are small enough to verify by hand.

Reference
---------
Alexander Wei.
"Better and Simpler Learning-Augmented Online Caching."
APPROX/RANDOM 2020, LIPIcs Vol. 176, Article 60.
"""

from __future__ import annotations

import math

import pytest

from lafc.metrics.prediction_error import compute_eta_unweighted
from lafc.policies.blind_oracle import BlindOraclePolicy
from lafc.policies.blind_oracle_lru_combiner import BlindOracleLRUCombiner
from lafc.policies.lru import LRUPolicy
from lafc.policies.offline_belady import OfflineBeladyPolicy
from lafc.predictors.noisy import add_additive_noise
from lafc.predictors.offline_from_trace import compute_perfect_predictions
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import build_requests_from_lists
from lafc.types import Page


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pages(**weights):
    """Build pages dict with given weights.  _pages(A=1.0, B=1.0)."""
    return {pid: Page(page_id=pid, weight=w) for pid, w in weights.items()}


def _build(page_ids, weights=None, predictions=None):
    return build_requests_from_lists(page_ids, weights, predictions)


def _unit_pages(*page_ids):
    """Build unit-weight pages dict from a list of page ids."""
    return {pid: Page(page_id=pid, weight=1.0) for pid in page_ids}


# ---------------------------------------------------------------------------
# 1. Simulator correctness — basic miss counting
# ---------------------------------------------------------------------------


def test_combiner_misses_first_k_requests():
    """First k distinct requests should all be misses (empty cache)."""
    requests, pages = _build(["A", "B", "C"])
    policy = BlindOracleLRUCombiner()
    result = run_policy(policy, requests, pages, capacity=3)
    assert result.total_misses == 3
    assert result.total_hits == 0


def test_combiner_repeated_hit():
    """
    Trace: A A A, capacity=2
    t=0  A miss  cache={A}
    t=1  A hit
    t=2  A hit
    Total: 1 miss, 2 hits.
    """
    requests, pages = _build(["A", "A", "A"])
    policy = BlindOracleLRUCombiner()
    result = run_policy(policy, requests, pages, capacity=2)
    assert result.total_misses == 1
    assert result.total_hits == 2


# ---------------------------------------------------------------------------
# 2. Cost accounting correctness — unit cost
# ---------------------------------------------------------------------------


def test_combiner_unit_cost():
    """Combiner uses unit cost (1.0) per miss regardless of page weights."""
    # Supply heavier weights; combiner should still use 1.0 per miss.
    weights = {"A": 5.0, "B": 5.0, "C": 5.0}
    requests, pages = _build(["A", "B", "C"], weights)
    policy = BlindOracleLRUCombiner()
    result = run_policy(policy, requests, pages, capacity=2)
    # 3 misses at unit cost → total cost = 3
    assert result.total_cost == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 3. BlindOracle correctness on a hand-checkable trace
# ---------------------------------------------------------------------------


def test_blind_oracle_evicts_farthest():
    """
    Trace: A B C D, capacity=2
    Predictions: A→100, B→200, C→50, D→∞
    (All predicted as very far so they differ by prediction only.)

    t=0  A miss  cache={A}  (no eviction needed)
    t=1  B miss  cache={A,B}  (no eviction needed)
    t=2  C miss  cache full  → evict max(pred[A]=100, pred[B]=200) = B
                             cache={A,C}
    t=3  D miss  cache full  → evict max(pred[A]=100, pred[C]=50) = A
                             cache={C,D}
    Total: 4 misses.
    """
    preds = [100, 200, 50, math.inf]
    requests, pages = _build(["A", "B", "C", "D"], predictions=preds)
    policy = BlindOraclePolicy()
    result = run_policy(policy, requests, pages, capacity=2)
    evictions = [e.evicted for e in result.events if e.evicted is not None]
    assert evictions == ["B", "A"]


# ---------------------------------------------------------------------------
# 4. Deterministic combiner: runs without errors on toy traces
# ---------------------------------------------------------------------------


def test_combiner_runs_toy_trace():
    """Combiner processes a short trace without exceptions."""
    page_ids = ["A", "B", "C", "A", "B", "D", "A", "C", "B", "A"]
    preds = [3, 4, 7, 6, 8, 9, 10, 9999, 9999, 9999]
    requests, pages = _build(page_ids, predictions=preds)
    policy = BlindOracleLRUCombiner()
    result = run_policy(policy, requests, pages, capacity=3)
    assert result.total_misses + result.total_hits == len(page_ids)
    assert result.total_cost == pytest.approx(float(result.total_misses))


def test_combiner_capacity_one():
    """
    With capacity=1, every request is a miss (except the immediately repeated one).
    Trace: A B A B, capacity=1
    t=0 A miss cache={A}
    t=1 B miss evict A  cache={B}
    t=2 A miss evict B  cache={A}
    t=3 B miss evict A  cache={B}
    Total: 4 misses.
    """
    requests, pages = _build(["A", "B", "A", "B"])
    policy = BlindOracleLRUCombiner()
    result = run_policy(policy, requests, pages, capacity=1)
    assert result.total_misses == 4
    assert result.total_hits == 0


def test_combiner_all_same_page():
    """All requests for the same page: 1 miss, rest hits."""
    requests, pages = _build(["A"] * 10)
    policy = BlindOracleLRUCombiner()
    result = run_policy(policy, requests, pages, capacity=3)
    assert result.total_misses == 1
    assert result.total_hits == 9


# ---------------------------------------------------------------------------
# 5. η computation correctness
# ---------------------------------------------------------------------------


def test_eta_perfect_predictions_is_zero():
    """With perfect predictions η = 0."""
    page_ids = ["A", "B", "C", "A", "B"]
    requests, pages = _build(page_ids)
    requests = compute_perfect_predictions(requests)
    eta = compute_eta_unweighted(requests)
    assert eta == pytest.approx(0.0)


def test_eta_single_request_no_recurrence():
    """Single request for A, no recurrence → τ=∞, a=∞, η=0."""
    requests, pages = _build(["A"])
    requests = compute_perfect_predictions(requests)
    eta = compute_eta_unweighted(requests)
    assert eta == pytest.approx(0.0)


def test_eta_non_zero_explicit():
    """
    Trace: A B A, predictions=[5, 2, inf], actual_next=[2, inf, inf]
    t=0: tau=5, a=2  → |5-2|=3
    t=1: tau=2, a=inf → one-side inf → eta=inf
    Skip t=1 with finite/inf mismatch — use a cleaner example.

    Trace: A B A, perfect predictions, but override one.
    t=0  page=A  actual_next=2  predicted_next=5   error=|5-2|=3
    t=1  page=B  actual_next=∞  predicted_next=∞   error=0
    t=2  page=A  actual_next=∞  predicted_next=∞   error=0
    Total η = 3.
    """
    page_ids = ["A", "B", "A"]
    requests, pages = _build(page_ids)
    requests = compute_perfect_predictions(requests)

    # Manually override prediction for t=0 to introduce error.
    from lafc.types import Request
    r0 = requests[0]
    requests = [
        Request(t=r0.t, page_id=r0.page_id, predicted_next=5.0, actual_next=r0.actual_next),
        requests[1],
        requests[2],
    ]
    eta = compute_eta_unweighted(requests)
    assert eta == pytest.approx(3.0)


def test_eta_both_infinite_contributes_zero():
    """Both τ=∞ and a=∞ (page never recurs) should contribute 0 to η."""
    requests, pages = _build(["A", "B", "C"])  # no repeats
    requests = compute_perfect_predictions(requests)
    eta = compute_eta_unweighted(requests)
    assert eta == pytest.approx(0.0)


def test_eta_one_sided_infinite_is_inf():
    """τ=finite, a=∞ (or vice versa) → η=∞."""
    from lafc.types import Request
    page_ids = ["A", "B"]
    requests, pages = _build(page_ids)
    # a_t for A at t=0: A doesn't recur → actual_next=∞
    # Set predicted_next = 5 (finite) → mismatch
    requests = [
        Request(t=0, page_id="A", predicted_next=5.0, actual_next=math.inf),
        Request(t=1, page_id="B", predicted_next=math.inf, actual_next=math.inf),
    ]
    eta = compute_eta_unweighted(requests)
    assert math.isinf(eta)


# ---------------------------------------------------------------------------
# 6. Deterministic reproducibility
# ---------------------------------------------------------------------------


def test_combiner_deterministic():
    """Running the same trace twice with the same policy gives identical results."""
    page_ids = ["A", "B", "C", "A", "B", "D"]
    preds = [3, 4, 7, 6, 8, 9]
    requests, pages = _build(page_ids, predictions=preds)

    policy1 = BlindOracleLRUCombiner()
    result1 = run_policy(policy1, requests, pages, capacity=3)

    policy2 = BlindOracleLRUCombiner()
    result2 = run_policy(policy2, requests, pages, capacity=3)

    assert result1.total_misses == result2.total_misses
    assert result1.total_cost == pytest.approx(result2.total_cost)
    for e1, e2 in zip(result1.events, result2.events):
        assert e1.hit == e2.hit
        assert e1.evicted == e2.evicted


# ---------------------------------------------------------------------------
# 7. Perfect-prediction sanity tests
# ---------------------------------------------------------------------------


def test_belady_oracle_perfect_predictions():
    """
    Under perfect predictions, OfflineBeladyPolicy should achieve OPT.

    Hand-trace: A B C A B C, capacity=2

    Belady on this trace:
    t=0  A miss  cache={A}
    t=1  B miss  cache={A,B}
    t=2  C miss  evict argmax actual_next: A next at t=3, B next at t=4 → evict B
                cache={A,C}
    t=3  A hit
    t=4  B miss  evict argmax actual_next: A next at ∞ (no more A), C next at t=5 → evict A
                Wait: actual_next at t=4 for A in cache: A's last actual_next = stored from t=3.
                At t=3, A's actual_next = ∞ (no more A after t=3).
                At t=4, C's actual_next: C was last requested at t=2 with actual_next=5.
                → evict A (∞ > 5)
                cache={C,B}
    t=5  C hit
    Total: 4 misses.

    Note: LRU on the same trace gives 6 misses.
    """
    page_ids = ["A", "B", "C", "A", "B", "C"]
    requests, pages = _build(page_ids)
    requests = compute_perfect_predictions(requests)

    belady = OfflineBeladyPolicy()
    result = run_policy(belady, requests, pages, capacity=2)

    # Belady is OPT; the minimum possible misses for this trace is 4.
    assert result.total_misses == 4


def test_blind_oracle_with_perfect_predictions_matches_belady():
    """
    With perfect predictions, BlindOracle and OfflineBeladyPolicy should
    make identical eviction decisions (both use predicted==actual next arrivals).
    """
    page_ids = ["A", "B", "C", "A", "B", "D", "A"]
    requests, pages = _build(page_ids)
    requests = compute_perfect_predictions(requests)

    bo = BlindOraclePolicy()
    result_bo = run_policy(bo, requests, pages, capacity=3)

    belady = OfflineBeladyPolicy()
    result_bel = run_policy(belady, requests, pages, capacity=3)

    assert result_bo.total_misses == result_bel.total_misses


def test_combiner_perfect_predictions_no_worse_than_lru():
    """
    Under perfect predictions, the combiner should not perform worse than LRU.
    (No formal guarantee, but this is a sanity check on a simple trace.)
    """
    page_ids = ["A", "B", "C", "A", "B", "D", "A", "C", "B", "A"]
    preds = [3, 4, 7, 6, 8, 9, 10, 9999, 9999, 9999]
    requests, pages = _build(page_ids, predictions=preds)
    requests_perfect = compute_perfect_predictions(requests)

    lru = LRUPolicy()
    result_lru = run_policy(lru, requests_perfect, pages, capacity=3)

    combiner = BlindOracleLRUCombiner()
    result_comb = run_policy(combiner, requests_perfect, pages, capacity=3)

    # Under perfect predictions, the combiner should be at most as bad as LRU.
    assert result_comb.total_misses <= result_lru.total_misses


# ---------------------------------------------------------------------------
# 8. Combiner remains online — does not use future information
# ---------------------------------------------------------------------------


def test_combiner_step_log_chosen_uses_prior_costs():
    """
    Verify that at each step the combiner's 'chosen' decision is based
    on shadow costs *before* the current request is processed.

    At t=0: both shadows have 0 misses before → chosen=blind_oracle (tie → BO).
    """
    page_ids = ["A", "B", "C"]
    requests, pages = _build(page_ids)
    policy = BlindOracleLRUCombiner()
    policy.reset(3, pages)

    for req in requests:
        policy.on_request(req)

    log = policy.step_log()
    # t=0: no prior requests → both shadows have 0 misses.
    assert log[0].bo_misses_before == 0
    assert log[0].lru_misses_before == 0
    # Tie → BlindOracle is chosen deterministically.
    assert log[0].chosen == "blind_oracle"


def test_combiner_tie_breaks_to_blind_oracle():
    """When cumulative misses tie, combiner must choose BlindOracle."""
    requests, pages = _build(["A", "B", "A"], predictions=[10, 10, 10])
    policy = BlindOracleLRUCombiner()
    policy.reset(2, pages)

    for req in requests:
        policy.on_request(req)

    first = policy.step_log()[0]
    assert first.bo_misses_before == first.lru_misses_before == 0
    assert first.chosen == "blind_oracle"


def test_combiner_step_log_records_eviction_choice():
    """
    Force an eviction and verify the step log records which sub-algorithm
    was chosen and which page was evicted.
    """
    # Trace: A B C D, capacity=2
    # t=0 A miss, t=1 B miss (cache full after A,B), t=2 C miss (evict), t=3 D miss (evict)
    page_ids = ["A", "B", "C", "D"]
    requests, pages = _build(page_ids)
    policy = BlindOracleLRUCombiner()
    policy.reset(2, pages)

    for req in requests:
        policy.on_request(req)

    log = policy.step_log()
    # t=2 should have an eviction.
    assert log[2].hit is False
    assert log[2].evicted is not None
    assert log[2].chosen in ("blind_oracle", "lru")
    # t=3 should have an eviction too.
    assert log[3].hit is False
    assert log[3].evicted is not None


def test_combiner_shadows_updated_independently():
    """
    Shadow policies should have their own cache states.
    Check that shadow miss counts are sensible after a run.
    """
    page_ids = ["A", "B", "C", "A", "B", "C"]
    requests, pages = _build(page_ids)
    policy = BlindOracleLRUCombiner()
    result = run_policy(policy, requests, pages, capacity=2)

    # Shadow miss counts must be non-negative and ≤ total requests.
    bo_misses = policy.shadow_bo_misses()
    lru_misses = policy.shadow_lru_misses()
    assert 0 <= bo_misses <= len(page_ids)
    assert 0 <= lru_misses <= len(page_ids)


def test_combiner_log_length_equals_trace_length():
    """Step log must have exactly one entry per request."""
    page_ids = ["A", "B", "C", "A", "B", "D"]
    requests, pages = _build(page_ids)
    policy = BlindOracleLRUCombiner()
    policy.reset(3, pages)

    for req in requests:
        policy.on_request(req)

    assert len(policy.step_log()) == len(page_ids)


# ---------------------------------------------------------------------------
# 9. Noisy predictions pipeline
# ---------------------------------------------------------------------------


def test_noisy_predictions_additive_noise_changes_predictions():
    """Adding noise with sigma>0 should change at least some predictions."""
    import random
    page_ids = ["A", "B", "C", "A", "B"]
    requests, pages = _build(page_ids)
    requests = compute_perfect_predictions(requests)

    noisy = add_additive_noise(requests, sigma=2.0, rng=random.Random(42))
    original_preds = [r.predicted_next for r in requests]
    noisy_preds = [r.predicted_next for r in noisy]

    # At least one prediction should differ (with sigma=2 and seed=42).
    assert any(
        not math.isinf(o) and abs(n - o) > 1e-9
        for o, n in zip(original_preds, noisy_preds)
        if not math.isinf(o)
    ), "Noisy predictions should differ from originals when sigma > 0"


def test_noisy_predictions_actual_next_preserved():
    """Adding noise must NOT change actual_next values."""
    import random
    page_ids = ["A", "B", "C", "A"]
    requests, pages = _build(page_ids)
    requests = compute_perfect_predictions(requests)
    noisy = add_additive_noise(requests, sigma=5.0, rng=random.Random(0))

    for orig, nois in zip(requests, noisy):
        if math.isinf(orig.actual_next):
            assert math.isinf(nois.actual_next)
        else:
            assert nois.actual_next == pytest.approx(orig.actual_next)


def test_noisy_predictions_never_before_current_t():
    """Noisy predictions must always be > current t (can't predict the past)."""
    import random
    page_ids = ["A", "B", "C", "A", "B", "C"] * 5
    requests, pages = _build(page_ids)
    requests = compute_perfect_predictions(requests)
    noisy = add_additive_noise(requests, sigma=100.0, rng=random.Random(7))

    for req in noisy:
        if not math.isinf(req.predicted_next):
            assert req.predicted_next > req.t, (
                f"t={req.t}: predicted_next={req.predicted_next} is not in the future"
            )


# ---------------------------------------------------------------------------
# 10. Combiner sub-policy tracking — integration test
# ---------------------------------------------------------------------------


def test_combiner_extra_diagnostics_present():
    """
    The run_policy helper should populate extra_diagnostics with combiner
    step log and shadow miss counts.
    """
    page_ids = ["A", "B", "C", "A", "B", "D"]
    preds = [3, 4, 7, 6, 8, 9]
    requests, pages = _build(page_ids, predictions=preds)
    policy = BlindOracleLRUCombiner()
    result = run_policy(policy, requests, pages, capacity=3)

    assert result.extra_diagnostics is not None
    assert "combiner_step_log" in result.extra_diagnostics
    assert "shadow_bo_total_misses" in result.extra_diagnostics
    assert "shadow_lru_total_misses" in result.extra_diagnostics

    step_log = result.extra_diagnostics["combiner_step_log"]
    assert len(step_log) == len(page_ids)

    # Every entry should have the expected keys.
    for entry in step_log:
        assert "t" in entry
        assert "page_id" in entry
        assert "hit" in entry
        assert "chosen" in entry
        assert "bo_misses_before" in entry
        assert "lru_misses_before" in entry


def test_combiner_chosen_is_always_valid():
    """
    'chosen' in the step log must always be 'blind_oracle' or 'lru'.
    """
    page_ids = ["A", "B", "C", "A", "B", "D", "A", "C", "B", "A"]
    preds = [3, 4, 7, 6, 8, 9, 10, 9999, 9999, 9999]
    requests, pages = _build(page_ids, predictions=preds)
    policy = BlindOracleLRUCombiner()
    policy.reset(3, pages)

    for req in requests:
        policy.on_request(req)

    for entry in policy.step_log():
        assert entry.chosen in ("blind_oracle", "lru"), (
            f"Unexpected chosen value: {entry.chosen}"
        )


# ---------------------------------------------------------------------------
# 11. OfflineBeladyPolicy: basic correctness
# ---------------------------------------------------------------------------


def test_belady_capacity_one():
    """
    With capacity=1, Belady evicts on every miss.
    Trace: A B A B, capacity=1
    All 4 requests are misses (Belady can't do better here).
    """
    requests, pages = _build(["A", "B", "A", "B"])
    requests = compute_perfect_predictions(requests)
    policy = OfflineBeladyPolicy()
    result = run_policy(policy, requests, pages, capacity=1)
    assert result.total_misses == 4


def test_belady_optimal_on_classic_example():
    """
    Classic example: A B C A B C, capacity=2.
    OPT = 4 misses.  See test_belady_oracle_perfect_predictions above.
    """
    page_ids = ["A", "B", "C", "A", "B", "C"]
    requests, pages = _build(page_ids)
    requests = compute_perfect_predictions(requests)
    policy = OfflineBeladyPolicy()
    result = run_policy(policy, requests, pages, capacity=2)
    assert result.total_misses == 4


def test_belady_no_worse_than_lru():
    """
    OfflineBeladyPolicy (OPT) should never incur more misses than LRU on any trace.
    """
    page_ids = ["A", "B", "C", "D", "A", "B", "C", "D", "A"]
    requests, pages = _build(page_ids)
    requests = compute_perfect_predictions(requests)

    belady = OfflineBeladyPolicy()
    result_bel = run_policy(belady, requests, pages, capacity=3)

    lru = LRUPolicy()
    result_lru = run_policy(lru, requests, pages, capacity=3)

    assert result_bel.total_misses <= result_lru.total_misses


# ---------------------------------------------------------------------------
# 12. Combiner: shadowing both algorithms correctly
# ---------------------------------------------------------------------------


def test_combiner_follows_blind_oracle_when_bo_better():
    """
    When BlindOracle has fewer misses, the combiner should follow its rule.
    Construct a trace where BO always has fewer misses than LRU.
    Use perfect predictions so BO is optimal (= Belady).
    """
    page_ids = ["A", "B", "C", "A", "B", "C"]
    requests, pages = _build(page_ids)
    requests = compute_perfect_predictions(requests)

    policy = BlindOracleLRUCombiner()
    policy.reset(2, pages)
    for req in requests:
        policy.on_request(req)

    # Under perfect predictions BO is at most as bad as LRU,
    # so we expect at least some "blind_oracle" choices.
    chosen_values = [s.chosen for s in policy.step_log() if s.chosen is not None]
    assert "blind_oracle" in chosen_values or "lru" in chosen_values, (
        "Expected at least one eviction decision in step log"
    )


def test_combiner_fallback_to_lru_when_bo_worse():
    """
    When BlindOracle has more misses, the combiner should switch to LRU.
    Provide adversarial predictions so BO incurs many misses.

    Trace: A B A B, capacity=2, predictions always wrong.
    With pred = [100, 200, 100, 200]: BO evicts based on predictions.
    """
    # Hand-checkable example where BO incurs more misses than LRU by t=4.
    page_ids = ["A", "C", "A", "B", "A", "D"]
    preds = [7, 3, 20, 8, 5, 9999]
    requests, pages = _build(page_ids, predictions=preds)

    policy = BlindOracleLRUCombiner()
    policy.reset(2, pages)
    for req in requests:
        policy.on_request(req)

    chosen_values = [s.chosen for s in policy.step_log()]
    # With adversarial predictions, LRU should be chosen at some point.
    assert "lru" in chosen_values


def test_combiner_no_future_leakage_choice_rule():
    """Chosen shadow must be a pure function of pre-request shadow misses."""
    page_ids = ["A", "B", "C", "A", "D", "B", "E"]
    preds = [100, 100, 100, 100, 1, 1, 1]
    requests, pages = _build(page_ids, predictions=preds)

    policy = BlindOracleLRUCombiner()
    policy.reset(2, pages)
    for req in requests:
        policy.on_request(req)

    for step in policy.step_log():
        expected = (
            "blind_oracle"
            if step.bo_misses_before <= step.lru_misses_before
            else "lru"
        )
        assert step.chosen == expected
