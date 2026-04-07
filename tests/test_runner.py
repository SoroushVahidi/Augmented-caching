"""
Tests for the runner: run_policy() and output-file generation.
"""

from __future__ import annotations

import csv
import json
import math
import os

import pytest

from lafc.policies.advice_trusting import AdviceTrustingPolicy
from lafc.policies.la_weighted_paging_deterministic import LAWeightedPagingDeterministic
from lafc.policies.lru import LRUPolicy
from lafc.policies.weighted_lru import WeightedLRUPolicy
from lafc.runner.run_policy import run_policy, save_results
from lafc.simulator.request_trace import build_requests_from_lists
from lafc.types import SimulationResult


def _build(page_ids, weights, predictions=None, capacity=2):
    requests, pages = build_requests_from_lists(page_ids, weights, predictions)
    return requests, pages


# ---------------------------------------------------------------------------
# run_policy correctness
# ---------------------------------------------------------------------------


def test_run_policy_returns_simulation_result():
    requests, pages = _build(["A", "B", "A"], {"A": 1.0, "B": 2.0})
    result = run_policy(LRUPolicy(), requests, pages, capacity=2)
    assert isinstance(result, SimulationResult)
    assert result.policy_name == "lru"


def test_run_policy_cost_equals_sum_of_missed_weights():
    """Total cost must equal sum of w_p for each cache miss."""
    weights = {"A": 1.0, "B": 3.0, "C": 5.0}
    page_ids = ["A", "B", "C", "A", "B", "C"]
    requests, pages = build_requests_from_lists(page_ids, weights)

    for PolicyClass in [LRUPolicy, WeightedLRUPolicy, AdviceTrustingPolicy, LAWeightedPagingDeterministic]:
        policy = PolicyClass()
        result = run_policy(policy, requests, pages, capacity=2)
        expected = sum(
            weights[e.page_id] for e in result.events if not e.hit
        )
        assert result.total_cost == pytest.approx(expected), f"Failed for {policy.name}"


def test_run_policy_hits_plus_misses_equals_trace_length():
    requests, pages = _build(["A", "B", "C", "A"], {"A": 1.0, "B": 1.0, "C": 1.0})
    result = run_policy(LRUPolicy(), requests, pages, capacity=2)
    assert result.total_hits + result.total_misses == len(requests)


def test_run_policy_events_length():
    page_ids = ["A", "B", "C", "D", "A"]
    requests, pages = _build(page_ids, {p: 1.0 for p in "ABCD"})
    result = run_policy(LRUPolicy(), requests, pages, capacity=3)
    assert len(result.events) == len(page_ids)


def test_run_policy_empty_trace_raises():
    with pytest.raises(ValueError):
        run_policy(LRUPolicy(), [], {}, capacity=2)


def test_run_policy_invalid_capacity_raises():
    requests, pages = _build(["A"], {"A": 1.0})
    with pytest.raises(ValueError):
        run_policy(LRUPolicy(), requests, pages, capacity=0)


def test_run_policy_all_policies_smoke():
    """All supported policies should run without errors on a basic trace."""
    weights = {"A": 1.0, "B": 2.0, "C": 4.0}
    page_ids = ["A", "B", "C", "A", "B", "C", "A"]
    requests, pages = build_requests_from_lists(page_ids, weights)

    for PolicyClass in [
        LRUPolicy,
        WeightedLRUPolicy,
        AdviceTrustingPolicy,
        LAWeightedPagingDeterministic,
    ]:
        result = run_policy(PolicyClass(), requests, pages, capacity=2)
        assert result.total_cost >= 0


def test_run_policy_prediction_error_eta_populated():
    """run_policy should populate prediction_error_eta in the result."""
    requests, pages = _build(["A", "B", "A"], {"A": 1.0, "B": 1.0})
    result = run_policy(LRUPolicy(), requests, pages, capacity=2)
    # eta should be populated (could be 0 or finite or inf)
    assert result.prediction_error_eta is not None


# ---------------------------------------------------------------------------
# save_results output files
# ---------------------------------------------------------------------------


def test_save_results_creates_files(tmp_path):
    requests, pages = _build(["A", "B"], {"A": 1.0, "B": 2.0})
    result = run_policy(LRUPolicy(), requests, pages, capacity=2)
    save_results(result, str(tmp_path))

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "per_step_decisions.csv").exists()


def test_save_results_summary_content(tmp_path):
    requests, pages = _build(["A", "B", "A"], {"A": 1.0, "B": 2.0})
    result = run_policy(LRUPolicy(), requests, pages, capacity=2)
    save_results(result, str(tmp_path))

    with open(tmp_path / "summary.json") as f:
        summary = json.load(f)

    assert summary["policy_name"] == "lru"
    assert "total_cost" in summary
    assert "hit_rate" in summary


def test_save_results_csv_row_count(tmp_path):
    page_ids = ["A", "B", "C"]
    requests, pages = _build(page_ids, {p: 1.0 for p in "ABC"})
    result = run_policy(LRUPolicy(), requests, pages, capacity=2)
    save_results(result, str(tmp_path))

    with open(tmp_path / "per_step_decisions.csv") as f:
        rows = list(csv.reader(f))

    # Header + one row per request.
    assert len(rows) == len(page_ids) + 1
