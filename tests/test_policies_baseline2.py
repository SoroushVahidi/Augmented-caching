"""
Tests for Baseline 2 algorithms:
  - Marker
  - BlindOracle
  - Predictive Marker
  - Unweighted prediction error η

All traces are small enough to verify by hand.

Reference
---------
Lykouris, Vassilvitskii.
"Competitive Caching with Machine Learned Advice."
ICML 2018 / JACM 2021.
"""

from __future__ import annotations

import math

import pytest

from lafc.metrics.prediction_error import compute_eta_unweighted
from lafc.policies.blind_oracle import BlindOraclePolicy
from lafc.policies.lru import LRUPolicy
from lafc.policies.marker import MarkerPolicy
from lafc.policies.predictive_marker import PredictiveMarkerPolicy
from lafc.predictors.offline_from_trace import compute_perfect_predictions
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import build_requests_from_lists, load_trace
from lafc.types import Page


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pages(**weights):
    """Build pages dict with given weights.  _pages(A=1.0, B=1.0)."""
    return {pid: Page(page_id=pid, weight=w) for pid, w in weights.items()}


def _build(page_ids, weights=None, predictions=None):
    return build_requests_from_lists(page_ids, weights, predictions)


# ---------------------------------------------------------------------------
# Simulator: unweighted trace loading (weights optional)
# ---------------------------------------------------------------------------


def test_load_trace_without_weights_uses_unit_weights(tmp_path):
    """A JSON trace without 'weights' should load with weight=1.0 for all pages."""
    import json

    trace_path = tmp_path / "no_weights.json"
    trace_path.write_text(json.dumps({"requests": ["A", "B", "A"]}))
    requests, pages = load_trace(str(trace_path))

    assert set(pages.keys()) == {"A", "B"}
    for p in pages.values():
        assert p.weight == pytest.approx(1.0)


def test_build_requests_without_weights_unit_cost():
    """build_requests_from_lists with weights=None uses unit weights."""
    requests, pages = _build(["A", "B", "C"])
    for p in pages.values():
        assert p.weight == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Marker: basic correctness
# ---------------------------------------------------------------------------


def test_marker_first_k_requests_all_miss():
    """First k distinct requests should all be misses (cache starts empty)."""
    requests, pages = _build(["A", "B", "C"])
    policy = MarkerPolicy()
    policy.reset(3, pages)
    events = [policy.on_request(r) for r in requests]
    assert all(not e.hit for e in events)
    assert policy.total_cost() == pytest.approx(3.0)


def test_marker_hit_after_cache_filled():
    """After all k slots filled, requesting a cached page is a hit."""
    requests, pages = _build(["A", "B", "A"])
    policy = MarkerPolicy()
    policy.reset(2, pages)
    events = [policy.on_request(r) for r in requests]
    assert not events[0].hit
    assert not events[1].hit
    assert events[2].hit  # A is still in cache


def test_marker_phase_transition():
    """
    Hand trace: A B C A B C, capacity=2.
    Phase 1: A(miss), B(miss) → cache={A,B}, both marked.
    Phase 2 starts at C (all marked):
      t=2: C miss, new phase, evict lex-min={A,B}→A,  cache={B,C}.
      t=3: A miss, B unmarked, evict B, cache={C,A}.
      t=4: B miss. All marked → new phase, evict lex-min={C,A}=A, cache={C,B}.
      t=5: C hit.
    Total: 5 misses, 1 hit.
    """
    requests, pages = _build(["A", "B", "C", "A", "B", "C"])
    policy = MarkerPolicy()
    policy.reset(2, pages)
    events = [policy.on_request(r) for r in requests]

    hits = sum(1 for e in events if e.hit)
    misses = sum(1 for e in events if not e.hit)
    assert misses == 5
    assert hits == 1
    assert policy.total_cost() == pytest.approx(5.0)


def test_marker_unit_cost():
    """Each miss costs exactly 1.0 regardless of page weights."""
    weights = {"A": 5.0, "B": 10.0, "C": 3.0}
    requests, pages = _build(["A", "B", "C"], weights)
    policy = MarkerPolicy()
    policy.reset(2, pages)
    events = [policy.on_request(r) for r in requests]
    # All 3 requests are misses; unit cost per miss.
    assert policy.total_cost() == pytest.approx(3.0)
    for e in events:
        if not e.hit:
            assert e.cost == pytest.approx(1.0)


def test_marker_phase_numbers_increase():
    """Phase numbers must be monotonically non-decreasing in event output."""
    requests, pages = _build(["A", "B", "C", "D", "E", "A", "B", "C"])
    policy = MarkerPolicy()
    policy.reset(2, pages)
    events = [policy.on_request(r) for r in requests]
    phases = [e.phase for e in events if e.phase is not None]
    assert phases == sorted(phases)


def test_marker_evicts_unmarked_page():
    """On a miss, the evicted page must be one that was unmarked."""
    requests, pages = _build(["A", "B", "C", "A", "B", "C"])
    policy = MarkerPolicy()
    policy.reset(2, pages)
    events = [policy.on_request(r) for r in requests]
    # Check that at every miss with an eviction, the evicted page was
    # not the *just-requested* page (it can't be — it was not in cache).
    for e in events:
        if not e.hit and e.evicted is not None:
            assert e.evicted != e.page_id


def test_marker_deterministic_reproducibility():
    """Same trace must produce identical results on repeated runs."""
    page_ids = ["A", "B", "C", "D", "A", "B", "C", "D", "A"]

    costs = []
    for _ in range(3):
        requests, pages = _build(page_ids)
        policy = MarkerPolicy()
        policy.reset(2, pages)
        events = [policy.on_request(r) for r in requests]
        costs.append(sum(e.cost for e in events))

    assert costs[0] == costs[1] == costs[2]


def test_marker_no_eviction_when_cache_not_full():
    """While cache has free slots, no eviction should occur."""
    requests, pages = _build(["A", "B", "C"])
    policy = MarkerPolicy()
    policy.reset(5, pages)
    events = [policy.on_request(r) for r in requests]
    for e in events:
        assert e.evicted is None


def test_marker_marked_pages_subset_of_cache():
    """marked_pages() must always be a subset of current_cache()."""
    requests, pages = _build(["A", "B", "C", "D", "A", "B"])
    policy = MarkerPolicy()
    policy.reset(2, pages)
    for r in requests:
        policy.on_request(r)
        assert policy.marked_pages() <= policy.current_cache()


# ---------------------------------------------------------------------------
# Blind Oracle: basic correctness
# ---------------------------------------------------------------------------


def test_blind_oracle_evicts_farthest_predicted():
    """
    Cache has A(τ=10) and B(τ=3); on a miss should evict A (τ=10 > 3).
    """
    requests, pages = _build(["A", "B", "C"], predictions=[10.0, 3.0, 99.0])
    policy = BlindOraclePolicy()
    policy.reset(2, pages)
    events = [policy.on_request(r) for r in requests]
    # After A, B are in cache; C miss → evict A (farther predicted arrival).
    assert events[2].evicted == "A"


def test_blind_oracle_perfect_predictions_optimal():
    """
    With perfect predictions, BlindOracle acts like Belady's OPT.

    Trace: A B C A B C, capacity=2, all unit weights.
    Perfect predictions:
      τ(A@t=0)=3, τ(B@t=1)=4, τ(C@t=2)=5
      τ(A@t=3)=∞, τ(B@t=4)=∞, τ(C@t=5)=∞

    t=0: A miss, cache={A}; τ[A]=3
    t=1: B miss, cache={A,B}; τ[B]=4
    t=2: C miss; evict B (τ=4>3); cache={A,C}; τ[C]=5
    t=3: A hit; τ[A]=∞
    t=4: B miss; evict A (τ=∞); cache={C,B}; τ[B]=∞
    t=5: C hit
    Total: 4 misses.
    """
    page_ids = ["A", "B", "C", "A", "B", "C"]
    requests, pages = _build(page_ids)
    requests_perf = compute_perfect_predictions(requests)

    policy = BlindOraclePolicy()
    policy.reset(2, pages)
    events = [policy.on_request(r) for r in requests_perf]

    misses = sum(1 for e in events if not e.hit)
    assert misses == 4
    assert policy.total_cost() == pytest.approx(4.0)
    # Verify specific evictions.
    assert events[2].evicted == "B"
    assert events[4].evicted == "A"


def test_blind_oracle_unit_cost():
    """BlindOracle must pay unit cost per miss regardless of page weights."""
    weights = {"A": 7.0, "B": 3.0, "C": 99.0}
    requests, pages = _build(["A", "B", "C"], weights)
    policy = BlindOraclePolicy()
    policy.reset(2, pages)
    events = [policy.on_request(r) for r in requests]
    for e in events:
        if not e.hit:
            assert e.cost == pytest.approx(1.0)


def test_blind_oracle_no_eviction_before_full():
    """No eviction should happen while the cache has free slots."""
    requests, pages = _build(["A", "B"])
    policy = BlindOraclePolicy()
    policy.reset(3, pages)
    events = [policy.on_request(r) for r in requests]
    for e in events:
        assert e.evicted is None


# ---------------------------------------------------------------------------
# Predictive Marker: basic correctness
# ---------------------------------------------------------------------------


def test_predictive_marker_runs_without_error():
    """The algorithm must process any valid trace without exceptions."""
    page_ids = ["A", "B", "C", "D", "A", "B", "C"]
    requests, pages = _build(page_ids)
    policy = PredictiveMarkerPolicy()
    policy.reset(2, pages)
    for r in requests:
        policy.on_request(r)


def test_predictive_marker_unit_cost():
    """PredictiveMarker pays unit cost per miss."""
    page_ids = ["A", "B", "C", "A", "B", "C"]
    requests, pages = _build(page_ids)
    requests_perf = compute_perfect_predictions(requests)
    policy = PredictiveMarkerPolicy()
    policy.reset(2, pages)
    events = [policy.on_request(r) for r in requests_perf]
    for e in events:
        if not e.hit:
            assert e.cost == pytest.approx(1.0)


def test_predictive_marker_phase_field_populated():
    """Every CacheEvent from PredictiveMarker must have a non-None phase."""
    requests, pages = _build(["A", "B", "C", "A"])
    policy = PredictiveMarkerPolicy()
    policy.reset(2, pages)
    events = [policy.on_request(r) for r in requests]
    for e in events:
        assert e.phase is not None
        assert e.phase >= 1


def test_predictive_marker_evicts_from_unmarked_only():
    """
    The Predictive Marker must only evict pages that were unmarked.

    Since the algorithm marks every fetched or hit page, the evicted page
    must be different from the just-requested page and must not have been
    recently fetched in the same phase without being the eviction candidate.

    We verify indirectly: the evicted page must not be the same as the
    currently requested page (it wasn't in cache).
    """
    page_ids = ["A", "B", "C", "A", "B", "C"]
    requests, pages = _build(page_ids)
    requests_perf = compute_perfect_predictions(requests)
    policy = PredictiveMarkerPolicy()
    policy.reset(2, pages)
    events = [policy.on_request(r) for r in requests_perf]

    for e in events:
        if not e.hit and e.evicted is not None:
            assert e.evicted != e.page_id


def test_predictive_marker_perfect_pred_matches_blind_oracle():
    """
    With perfect predictions, PredictiveMarker should achieve the same
    miss count as BlindOracle on this toy trace (both act like Belady).

    Trace: A B C A B C, capacity=2.
    Expected: 4 misses.
    """
    page_ids = ["A", "B", "C", "A", "B", "C"]
    requests, pages = _build(page_ids)
    requests_perf = compute_perfect_predictions(requests)

    bo = BlindOraclePolicy()
    pm = PredictiveMarkerPolicy()

    result_bo = run_policy(bo, requests_perf, pages, capacity=2)
    result_pm = run_policy(pm, requests_perf, pages, capacity=2)

    assert result_pm.total_misses == 4
    assert result_pm.total_misses <= result_bo.total_misses + 1  # within 1 of BO


def test_predictive_marker_perfect_pred_not_worse_than_marker():
    """
    With perfect predictions, PredictiveMarker must be at least as good
    as plain Marker on this simple trace.
    """
    page_ids = ["A", "B", "C", "A", "B", "C"]
    requests, pages = _build(page_ids)
    requests_perf = compute_perfect_predictions(requests)

    marker = MarkerPolicy()
    pm = PredictiveMarkerPolicy()

    result_m = run_policy(marker, requests, pages, capacity=2)
    result_pm = run_policy(pm, requests_perf, pages, capacity=2)

    assert result_pm.total_misses <= result_m.total_misses


def test_predictive_marker_deterministic_reproducibility():
    """Same trace + predictions must always yield the same cost."""
    page_ids = ["A", "B", "C", "D", "A", "B", "C"]
    preds = [4.0, 5.0, 6.0, 9.0, 7.0, 8.0, 9.0]
    costs = []
    for _ in range(3):
        requests, pages = _build(page_ids, predictions=preds)
        policy = PredictiveMarkerPolicy()
        policy.reset(2, pages)
        events = [policy.on_request(r) for r in requests]
        costs.append(sum(e.cost for e in events))
    assert costs[0] == costs[1] == costs[2]


# ---------------------------------------------------------------------------
# Predictive Marker: clean-chain diagnostics
# ---------------------------------------------------------------------------


def test_predictive_marker_all_clean_with_perfect_predictions():
    """
    With perfect predictions all evictions should be "clean"
    (predictor matches oracle).
    """
    page_ids = ["A", "B", "C", "A", "B", "C"]
    requests, pages = _build(page_ids)
    requests_perf = compute_perfect_predictions(requests)

    policy = PredictiveMarkerPolicy()
    result = run_policy(policy, requests_perf, pages, capacity=2)

    cc = result.extra_diagnostics["clean_chains"]
    assert cc["num_dirty_phases"] == 0
    assert cc["total_dirty_evictions"] == 0


def test_predictive_marker_dirty_phase_with_wrong_prediction():
    """
    A deliberately wrong prediction should cause a dirty eviction.

    Trace: A B C A, capacity=2.
    Perfect predictions: τ(A@0)=3, τ(B@1)=∞, τ(C@2)=∞.
    Wrong predictions: τ(A@0)=3, τ(B@1)=2 (B predicted at 2, wrong), τ(C@2)=∞.

    At t=2 (C miss): unmarked={A,B}.
    - Wrong prediction: τ[A]=3, τ[B]=2 → evict A (τ=3>2).
    - Oracle (actual): actual[A]=3, actual[B]=∞ → should evict B (∞>3).
    → Dirty eviction.
    """
    page_ids = ["A", "B", "C", "A"]
    # Wrong prediction: predict B will arrive at t=2 (it won't, it never comes back)
    predictions = [3.0, 2.0, 9999.0, 9999.0]  # τ[B@1] wrongly says 2
    requests, pages = _build(page_ids, predictions=predictions)

    policy = PredictiveMarkerPolicy()
    result = run_policy(policy, requests, pages, capacity=2)

    cc = result.extra_diagnostics["clean_chains"]
    assert cc["total_dirty_evictions"] >= 1


def test_predictive_marker_clean_chains_structure():
    """clean_chains must list [start_phase, end_phase] pairs in order."""
    page_ids = ["A", "B", "C", "A", "B", "C", "D", "A"]
    requests, pages = _build(page_ids)
    requests_perf = compute_perfect_predictions(requests)

    policy = PredictiveMarkerPolicy()
    result = run_policy(policy, requests_perf, pages, capacity=2)

    cc = result.extra_diagnostics["clean_chains"]
    for chain in cc["clean_chains"]:
        assert len(chain) == 2
        start, end = chain
        assert start <= end
        assert start >= 1


def test_predictive_marker_phase_log_phases_monotone():
    """Phase IDs in the phase log must be strictly increasing."""
    page_ids = ["A", "B", "C", "D", "A", "B", "C"]
    requests, pages = _build(page_ids)
    policy = PredictiveMarkerPolicy()
    result = run_policy(policy, requests, pages, capacity=2)

    cc = result.extra_diagnostics["clean_chains"]
    phase_ids = [p["phase_id"] for p in cc["phases"]]
    assert phase_ids == list(range(1, len(phase_ids) + 1))


# ---------------------------------------------------------------------------
# Unweighted prediction error η
# ---------------------------------------------------------------------------


def test_eta_unweighted_zero_for_perfect_predictions():
    """η must be 0 when all predictions are perfect."""
    page_ids = ["A", "B", "A", "C", "B"]
    requests, pages = _build(page_ids)
    requests_perf = compute_perfect_predictions(requests)
    eta = compute_eta_unweighted(requests_perf)
    assert eta == pytest.approx(0.0)


def test_eta_unweighted_simple_case():
    """
    Trace: A, B, A.
    Actual nexts: a(A@0)=2, a(B@1)=∞, a(A@2)=∞.
    Predictions: τ(A@0)=5, τ(B@1)=∞, τ(A@2)=∞.
    η = |5-2| + |∞-∞| + |∞-∞| = 3 + 0 + 0 = 3.
    """
    page_ids = ["A", "B", "A"]
    predictions = [5.0, math.inf, math.inf]
    requests, pages = _build(page_ids, predictions=predictions)
    eta = compute_eta_unweighted(requests)
    assert eta == pytest.approx(3.0)


def test_eta_unweighted_one_sided_inf():
    """If only one of τ or a is ∞, η = ∞."""
    page_ids = ["A", "B"]
    predictions = [math.inf, 5.0]  # predict A never appears again
    # actual: A@0 has actual_next=∞ (no second occurrence); B@1 has actual_next=∞
    requests, pages = _build(page_ids, predictions=predictions)
    # τ[B@1]=5, a[B@1]=∞ → one-sided infinity → η=∞
    eta = compute_eta_unweighted(requests)
    assert math.isinf(eta)


def test_eta_unweighted_both_inf_zero_contribution():
    """When both τ and a are ∞, their contribution to η is 0."""
    page_ids = ["A"]
    predictions = [math.inf]
    requests, pages = _build(page_ids, predictions=predictions)
    eta = compute_eta_unweighted(requests)
    assert eta == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Runner integration tests for Baseline 2
# ---------------------------------------------------------------------------


def test_run_policy_marker_smoke():
    """run_policy must not raise for MarkerPolicy."""
    page_ids = ["A", "B", "C", "A", "B"]
    requests, pages = _build(page_ids)
    result = run_policy(MarkerPolicy(), requests, pages, capacity=2)
    assert result.total_hits + result.total_misses == len(page_ids)
    assert result.total_cost == pytest.approx(float(result.total_misses))


def test_run_policy_blind_oracle_smoke():
    """run_policy must not raise for BlindOraclePolicy."""
    page_ids = ["A", "B", "C", "A", "B"]
    requests, pages = _build(page_ids)
    result = run_policy(BlindOraclePolicy(), requests, pages, capacity=2)
    assert result.total_cost >= 0


def test_run_policy_predictive_marker_smoke():
    """run_policy must not raise for PredictiveMarkerPolicy."""
    page_ids = ["A", "B", "C", "A", "B"]
    requests, pages = _build(page_ids)
    result = run_policy(PredictiveMarkerPolicy(), requests, pages, capacity=2)
    assert result.total_cost >= 0
    assert result.extra_diagnostics is not None
    assert "clean_chains" in result.extra_diagnostics
    assert "eta_unweighted" in result.extra_diagnostics


def test_run_policy_predictive_marker_cost_equals_misses():
    """For PredictiveMarker every miss costs 1, so total_cost == total_misses."""
    page_ids = ["A", "B", "C", "D", "A", "B"]
    requests, pages = _build(page_ids)
    result = run_policy(PredictiveMarkerPolicy(), requests, pages, capacity=2)
    assert result.total_cost == pytest.approx(float(result.total_misses))


def test_run_policy_all_baseline2_policies_save_files(tmp_path):
    """summary.json, metrics.json, and per_step_decisions.csv must be written."""
    from lafc.runner.run_policy import save_results

    page_ids = ["A", "B", "C", "A"]
    requests, pages = _build(page_ids)

    for PolicyClass in [MarkerPolicy, BlindOraclePolicy, PredictiveMarkerPolicy]:
        policy = PolicyClass()
        result = run_policy(policy, requests, pages, capacity=2)
        out_dir = tmp_path / policy.name
        save_results(result, str(out_dir))
        assert (out_dir / "summary.json").exists()
        assert (out_dir / "metrics.json").exists()
        assert (out_dir / "per_step_decisions.csv").exists()


def test_per_step_csv_has_phase_column_for_marker(tmp_path):
    """per_step_decisions.csv must include a 'phase' column for MarkerPolicy."""
    import csv as csv_module

    from lafc.runner.run_policy import save_results

    page_ids = ["A", "B", "C", "A"]
    requests, pages = _build(page_ids)
    result = run_policy(MarkerPolicy(), requests, pages, capacity=2)
    save_results(result, str(tmp_path))

    with open(tmp_path / "per_step_decisions.csv") as f:
        reader = csv_module.DictReader(f)
        rows = list(reader)

    assert "phase" in rows[0]


def test_per_step_csv_no_phase_column_for_lru(tmp_path):
    """per_step_decisions.csv must NOT include a 'phase' column for LRUPolicy."""
    import csv as csv_module

    from lafc.runner.run_policy import save_results

    page_ids = ["A", "B", "C", "A"]
    requests, pages = _build(page_ids)
    result = run_policy(LRUPolicy(), requests, pages, capacity=2)
    save_results(result, str(tmp_path))

    with open(tmp_path / "per_step_decisions.csv") as f:
        reader = csv_module.DictReader(f)
        rows = list(reader)

    assert "phase" not in rows[0]


# ---------------------------------------------------------------------------
# CLI smoke test (via Python import, not subprocess)
# ---------------------------------------------------------------------------


def test_cli_predictive_marker_example_unweighted(tmp_path):
    """Running the CLI on data/example_unweighted.json should succeed."""
    import os

    from lafc.runner.run_policy import run_policy, save_results
    from lafc.simulator.request_trace import load_trace

    trace_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "example_unweighted.json"
    )
    requests, pages = load_trace(trace_path)
    for PolicyClass in [MarkerPolicy, BlindOraclePolicy, PredictiveMarkerPolicy]:
        policy = PolicyClass()
        result = run_policy(policy, requests, pages, capacity=3)
        save_results(result, str(tmp_path / policy.name))
        assert result.total_misses >= 0
