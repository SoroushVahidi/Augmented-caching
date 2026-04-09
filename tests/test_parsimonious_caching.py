"""Tests for AdaptiveQuery-b (Im et al., 2022 parsimonious caching baseline)."""

from __future__ import annotations

import pytest

from lafc.policies.adaptive_query import AdaptiveQueryPolicy
from lafc.runner.run_policy import POLICY_REGISTRY, run_policy
from lafc.simulator.request_trace import build_requests_from_lists


def _build(page_ids, predictions=None):
    return build_requests_from_lists(page_ids, weights=None, predictions=predictions)


def test_adaptive_query_runs_end_to_end_and_emits_diagnostics():
    requests, pages = _build(["A", "B", "C", "A", "D", "B", "E", "A"])
    result = run_policy(AdaptiveQueryPolicy(b=2, seed=7), requests, pages, capacity=3)

    assert result.total_hits + result.total_misses == len(requests)
    assert result.extra_diagnostics is not None
    assert "adaptive_query" in result.extra_diagnostics
    summary = result.extra_diagnostics["adaptive_query"]["summary"]
    assert "queries_used" in summary
    assert "fraction_misses_queried" in summary


def test_adaptive_query_registered_with_aliases():
    assert "adaptive_query" in POLICY_REGISTRY
    assert "parsimonious_caching" in POLICY_REGISTRY


def test_adaptive_query_is_parsimonious_not_query_all_misses():
    requests, pages = _build(["A", "B", "C", "D", "A", "B", "E", "A", "B", "C"])
    result = run_policy(AdaptiveQueryPolicy(b=2, seed=0), requests, pages, capacity=3)

    misses = result.total_misses
    q_used = result.extra_diagnostics["adaptive_query"]["summary"]["queries_used"]
    # If the policy queried every cached page on every miss once full, this
    # would be around misses * capacity; with b=2 it should be much smaller.
    assert q_used <= 2 * misses


def test_adaptive_query_respects_capacity_and_returns_valid_evictions():
    requests, pages = _build(["A", "B", "C", "A", "D", "E", "A", "B", "C", "D"])
    policy = AdaptiveQueryPolicy(b=2, seed=1)
    policy.reset(3, pages)

    for req in requests:
        event = policy.on_request(req)
        assert len(policy.current_cache()) <= 3
        if event.evicted is not None:
            assert event.evicted != req.page_id


def test_adaptive_query_seed_reproducibility():
    page_ids = ["A", "B", "C", "D", "A", "E", "B", "F", "A", "B", "C", "D"]
    requests, pages = _build(page_ids)

    r1 = run_policy(AdaptiveQueryPolicy(b=2, seed=123), requests, pages, capacity=3)
    r2 = run_policy(AdaptiveQueryPolicy(b=2, seed=123), requests, pages, capacity=3)

    assert r1.total_misses == r2.total_misses
    assert [e.evicted for e in r1.events] == [e.evicted for e in r2.events]


def test_adaptive_query_b1_uses_one_query_per_query_mode_eviction():
    requests, pages = _build(["A", "B", "C", "D", "E", "A", "B", "C", "D", "E"])
    result = run_policy(AdaptiveQueryPolicy(b=1, seed=9), requests, pages, capacity=3)

    summary = result.extra_diagnostics["adaptive_query"]["summary"]
    queried_misses = summary["query_mode_evictions"]
    queries_used = summary["queries_used"]
    assert queries_used == pytest.approx(queried_misses)
