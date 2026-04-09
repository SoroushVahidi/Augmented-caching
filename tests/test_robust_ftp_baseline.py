"""Tests for RobustFtP-D (Chłędowski et al. 2021 experimental baseline)."""

from __future__ import annotations

from lafc.policies.robust_ftp_marker_combiner import RobustFtPDeterministicMarkerCombiner
from lafc.predictors.offline_from_trace import attach_predicted_caches
from lafc.runner.run_policy import POLICY_REGISTRY, run_policy
from lafc.simulator.request_trace import build_requests_from_lists


def test_robust_ftp_registered_aliases():
    assert "robust_ftp_d_marker" in POLICY_REGISTRY
    assert "robust_ftp" in POLICY_REGISTRY


def test_robust_ftp_runs_end_to_end_with_predicted_caches():
    req, pages = build_requests_from_lists(["A", "B", "C", "A", "D", "B", "A"])
    req = attach_predicted_caches(req, capacity=3)

    result = run_policy(RobustFtPDeterministicMarkerCombiner(), req, pages, capacity=3)
    assert result.total_hits + result.total_misses == len(req)
    assert result.extra_diagnostics is not None
    assert "robust_ftp" in result.extra_diagnostics


def test_robust_ftp_switches_on_toy_trace():
    page_ids = ["A", "B", "C", "A", "B", "C", "A", "B", "C"]
    predicted_caches = [
        ["A", "B"],
        ["B"],
        [],
        [],
        ["A"],
        ["C"],
        [],
        ["A"],
        ["B"],
    ]
    req, pages = build_requests_from_lists(page_ids, predicted_caches=predicted_caches)

    result = run_policy(RobustFtPDeterministicMarkerCombiner(), req, pages, capacity=2)
    robust = result.extra_diagnostics["robust_ftp"]
    summary = robust["summary"]

    assert summary["switch_count"] >= 1
    assert robust["switch_points"]
    assert any(s["chosen_expert"] == "robust" for s in robust["step_log"])


def test_robust_ftp_deterministic_reproducibility():
    req, pages = build_requests_from_lists(["A", "B", "C", "D", "A", "B", "E", "A", "D"])
    req = attach_predicted_caches(req, capacity=3)

    r1 = run_policy(RobustFtPDeterministicMarkerCombiner(), req, pages, capacity=3)
    r2 = run_policy(RobustFtPDeterministicMarkerCombiner(), req, pages, capacity=3)

    assert r1.total_misses == r2.total_misses
    assert [e.evicted for e in r1.events] == [e.evicted for e in r2.events]


def test_robust_ftp_capacity_invariants():
    req, pages = build_requests_from_lists(["A", "B", "C", "A", "D", "E", "A", "B"])
    req = attach_predicted_caches(req, capacity=3)

    pol = RobustFtPDeterministicMarkerCombiner()
    pol.reset(3, pages)
    for r in req:
        ev = pol.on_request(r)
        assert len(pol.current_cache()) <= 3
        if ev.evicted is not None:
            assert ev.evicted != r.page_id
