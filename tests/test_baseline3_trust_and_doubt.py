from __future__ import annotations

import json

from lafc.metrics.prediction_error import compute_cache_state_error
from lafc.policies.trust_and_doubt import TrustAndDoubtPolicy
from lafc.predictors.offline_from_trace import attach_predicted_caches
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import build_requests_from_lists, load_trace


def _build_with_predicted_caches(page_ids, capacity, predicted_caches):
    req, pages = build_requests_from_lists(page_ids, predicted_caches=predicted_caches)
    return req, pages


def test_trust_and_doubt_smoke():
    requests, pages = build_requests_from_lists(["A", "B", "C", "A", "B", "D", "A"])
    requests = attach_predicted_caches(requests, capacity=3)
    result = run_policy(TrustAndDoubtPolicy(seed=1), requests, pages, capacity=3)
    assert result.total_misses >= 0
    assert len(result.events) == len(requests)


def test_trust_and_doubt_seed_reproducible():
    req, pages = build_requests_from_lists(["A", "B", "C", "A", "D", "B", "C", "D"])
    req = attach_predicted_caches(req, capacity=3)
    r1 = run_policy(TrustAndDoubtPolicy(seed=7), req, pages, capacity=3)
    r2 = run_policy(TrustAndDoubtPolicy(seed=7), req, pages, capacity=3)
    assert [e.hit for e in r1.events] == [e.hit for e in r2.events]


def test_trust_and_doubt_seed_changes_randomized_paths():
    # Hand-found trace where arbitrary randomized branches diverge by seed.
    req, pages = build_requests_from_lists(["A", "C", "G", "E", "H", "G", "F", "H"])
    req = attach_predicted_caches(req, capacity=3)
    r1 = run_policy(TrustAndDoubtPolicy(seed=1), req, pages, capacity=3)
    r2 = run_policy(TrustAndDoubtPolicy(seed=2), req, pages, capacity=3)
    # Randomized algorithm may differ by seed; require that at least one step changes.
    assert any((e1.hit, e1.evicted) != (e2.hit, e2.evicted) for e1, e2 in zip(r1.events, r2.events))


def test_cache_state_error_metric_present():
    req, _ = build_requests_from_lists(["A", "B", "C", "A"])
    req = attach_predicted_caches(req, capacity=2)
    err = compute_cache_state_error(req, capacity=2)
    assert err["total_error"] is not None
    assert len(err["per_step"]) == 4


def test_load_csv_trace_with_predicted_cache(tmp_path):
    p = tmp_path / "trace.csv"
    p.write_text("t,page_id,predicted_next,predicted_cache\n0,A,3,A|B\n1,B,4,A|B\n2,C,10,B|C\n")
    requests, pages = load_trace(str(p))
    assert requests[0].metadata["predicted_cache"] == ["A", "B"]
    assert pages["A"].weight == 1.0


def test_load_json_predicted_caches(tmp_path):
    p = tmp_path / "trace.json"
    p.write_text(json.dumps({
        "requests": ["A", "B", "C"],
        "predicted_caches": [["A"], ["A", "B"], ["B", "C"]],
    }))
    requests, _ = load_trace(str(p))
    assert requests[2].metadata["predicted_cache"] == ["B", "C"]


def test_clean_page_initializes_pq_trust_and_threshold():
    page_ids = ["A", "B", "C", "D"]
    pred = [["A", "B", "C"]] * len(page_ids)
    req, pages = _build_with_predicted_caches(page_ids, 3, pred)

    pol = TrustAndDoubtPolicy(seed=0)
    pol.reset(3, pages)
    ev = None
    for r in req:
        ev = pol.on_request(r)

    diag = ev.diagnostics
    assert "D" in diag["C"]
    assert diag["trusted"].get("D") is True
    assert diag["tq"].get("D", 0) >= 1
    assert "D" in diag["pq"]


def test_prediction_failure_sets_doubt_then_doubles_on_next_interval():
    # Build a trace that creates a clean page D then requests p_D, triggering doubt,
    # then crosses next q-interval boundary to force doubling.
    page_ids = ["A", "B", "C", "D", "A", "E", "F"]
    pred = [["A", "B", "C"] for _ in page_ids]
    req, pages = _build_with_predicted_caches(page_ids, 3, pred)

    pol = TrustAndDoubtPolicy(seed=0)
    pol.reset(3, pages)

    # Create clean D at t=3.
    for i in range(4):
        ev = pol.on_request(req[i])
    d1 = ev.diagnostics
    pq_d = d1["pq"]["D"]
    assert d1["trusted"]["D"] is True
    tq_before = d1["tq"]["D"]

    # Request p_D -> failure -> trusted(D) becomes False.
    fail_req = req[4]
    fail_req = type(fail_req)(
        t=fail_req.t,
        page_id=pq_d,
        predicted_next=fail_req.predicted_next,
        actual_next=fail_req.actual_next,
        metadata=fail_req.metadata,
    )
    ev_fail = pol.on_request(fail_req)
    d2 = ev_fail.diagnostics
    assert d2["trusted"]["D"] is False

    # Advance arrivals to next q-interval start.
    pol.on_request(req[5])
    ev_next = pol.on_request(req[6])
    d3 = ev_next.diagnostics
    assert d3["trusted"]["D"] is True
    assert d3["tq"]["D"] >= tq_before


def test_ancient_state_exposed_in_diagnostics():
    page_ids = ["A", "B", "C", "D", "A", "E"]
    pred = [["A", "B", "C"] for _ in page_ids]
    req, pages = _build_with_predicted_caches(page_ids, 3, pred)

    pol = TrustAndDoubtPolicy(seed=3)
    pol.reset(3, pages)
    events = [pol.on_request(r) for r in req]

    # At minimum, the state is exposed and typed as a list per event.
    assert all(isinstance(e.diagnostics["A"], list) for e in events)


def test_no_future_leakage_in_decisions():
    # Prefix-equivalent requests must produce identical decisions on the prefix.
    prefix = ["A", "B", "C", "D", "A", "E"]
    suffix1 = ["F", "G", "H"]
    suffix2 = ["X", "Y", "Z"]

    req1, pages1 = build_requests_from_lists(prefix + suffix1)
    req2, pages2 = build_requests_from_lists(prefix + suffix2)
    req1 = attach_predicted_caches(req1, capacity=3)
    req2 = attach_predicted_caches(req2, capacity=3)

    r1 = run_policy(TrustAndDoubtPolicy(seed=11), req1, pages1, capacity=3)
    r2 = run_policy(TrustAndDoubtPolicy(seed=11), req2, pages2, capacity=3)

    for e1, e2 in zip(r1.events[: len(prefix)], r2.events[: len(prefix)]):
        assert (e1.hit, e1.evicted) == (e2.hit, e2.evicted)


def test_runner_integration_with_derived_predicted_caches():
    req, pages = build_requests_from_lists(["A", "B", "C", "A", "D", "B"])
    req = attach_predicted_caches(req, capacity=3)
    res = run_policy(TrustAndDoubtPolicy(seed=0), req, pages, capacity=3)
    assert res.extra_diagnostics is not None
    assert "eta_unweighted" in res.extra_diagnostics
