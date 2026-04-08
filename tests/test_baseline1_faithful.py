from __future__ import annotations

import pytest

from lafc.policies.la_weighted_paging_det_faithful import LAWeightedPagingDeterministicFaithful
from lafc.policies.la_weighted_paging_deterministic import LAWeightedPagingDeterministic
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import build_requests_from_lists


def test_faithful_groups_pages_by_weight_class():
    req, pages = build_requests_from_lists(["A", "B", "C"], {"A": 1.0, "B": 1.0, "C": 4.0})
    pol = LAWeightedPagingDeterministicFaithful()
    pol.reset(2, pages)
    assert len(pol._classes) == 2
    weights = sorted(c.weight for c in pol._classes)
    assert weights == [1.0, 4.0]


def test_faithful_within_class_ranking_uses_predictions_only_within_class():
    req, pages = build_requests_from_lists(["A", "B", "C"], {"A": 1.0, "B": 1.0, "C": 4.0})
    pol = LAWeightedPagingDeterministicFaithful()
    pol.reset(2, pages)
    pol._predicted_next["A"] = 9.0
    pol._predicted_next["B"] = 3.0
    ci = pol._page_to_class["A"]
    assert pol._class_ranking(ci) == ["B", "A"]


def test_faithful_mass_conservation_sum_x_equals_k_plus_ell():
    req, pages = build_requests_from_lists(["A", "B", "C", "D", "A"], {"A": 1.0, "B": 2.0, "C": 4.0, "D": 8.0})
    pol = LAWeightedPagingDeterministicFaithful()
    pol.reset(3, pages)
    for r in req:
        pol.on_request(r)
        total_x = sum(c.x for c in pol._classes)
        assert total_x == pytest.approx(pol._k_eff, abs=1e-3)


def test_faithful_interval_measure_matches_mu_snapshot():
    req, pages = build_requests_from_lists(["A", "B", "C", "A", "D", "B"], {"A": 1.0, "B": 2.0, "C": 4.0, "D": 8.0})
    pol = LAWeightedPagingDeterministicFaithful()
    pol.reset(3, pages)
    ev = None
    for r in req:
        ev = pol.on_request(r)
    snap = ev.diagnostics["x_mu"]
    for c in pol._classes:
        cls = snap[str(c.weight)]
        assert cls["mu"] == pytest.approx(c.s.measure())


def test_faithful_deterministic_reproducibility():
    req, pages = build_requests_from_lists(["A", "B", "C", "D", "A", "B", "E"], {"A": 1.0, "B": 2.0, "C": 4.0, "D": 8.0, "E": 16.0})
    r1 = run_policy(LAWeightedPagingDeterministicFaithful(), req, pages, capacity=3)
    r2 = run_policy(LAWeightedPagingDeterministicFaithful(), req, pages, capacity=3)
    assert [(e.hit, e.evicted) for e in r1.events] == [(e.hit, e.evicted) for e in r2.events]


def test_faithful_smoke_small_weighted_example():
    req, pages = build_requests_from_lists(["A", "B", "C", "A", "D"], {"A": 1.0, "B": 2.0, "C": 4.0, "D": 8.0})
    res = run_policy(LAWeightedPagingDeterministicFaithful(), req, pages, capacity=2)
    assert res.total_hits + res.total_misses == len(req)


def test_faithful_and_old_heuristic_are_distinct_policies():
    req, pages = build_requests_from_lists(
        ["A", "B", "C", "A", "B", "D", "A"],
        {"A": 1.0, "B": 2.0, "C": 4.0, "D": 8.0},
        predictions=[6.0, 7.0, 50.0, 10.0, 11.0, 60.0, 99.0],
    )
    old_res = run_policy(LAWeightedPagingDeterministic(), req, pages, capacity=2)
    new_res = run_policy(LAWeightedPagingDeterministicFaithful(), req, pages, capacity=2)
    assert old_res.policy_name != new_res.policy_name


def test_faithful_no_future_leakage_on_prefix():
    prefix = ["A", "B", "C", "A", "D"]
    req1, pages1 = build_requests_from_lists(prefix + ["E", "F"], {"A": 1.0, "B": 2.0, "C": 4.0, "D": 8.0, "E": 16.0, "F": 32.0})
    req2, pages2 = build_requests_from_lists(prefix + ["X", "Y"], {"A": 1.0, "B": 2.0, "C": 4.0, "D": 8.0, "X": 16.0, "Y": 32.0})
    r1 = run_policy(LAWeightedPagingDeterministicFaithful(), req1, pages1, capacity=2)
    r2 = run_policy(LAWeightedPagingDeterministicFaithful(), req2, pages2, capacity=2)
    for a, b in zip(r1.events[: len(prefix)], r2.events[: len(prefix)]):
        assert (a.hit, a.evicted) == (b.hit, b.evicted)
