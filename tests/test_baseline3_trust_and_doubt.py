from __future__ import annotations

import json

from lafc.metrics.prediction_error import compute_cache_state_error
from lafc.policies.trust_and_doubt import TrustAndDoubtPolicy
from lafc.predictors.offline_from_trace import attach_predicted_caches
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import build_requests_from_lists, load_trace


def test_trust_and_doubt_smoke():
    requests, pages = build_requests_from_lists(["A", "B", "C", "A", "B", "D", "A"])
    requests = attach_predicted_caches(requests, capacity=3)
    result = run_policy(TrustAndDoubtPolicy(seed=1), requests, pages, capacity=3)
    assert result.total_misses >= 0
    assert len(result.events) == len(requests)


def test_trust_and_doubt_deterministic_seed():
    req, pages = build_requests_from_lists(["A", "B", "C", "A", "D", "B", "C", "D"])
    req = attach_predicted_caches(req, capacity=3)
    r1 = run_policy(TrustAndDoubtPolicy(seed=7), req, pages, capacity=3)
    r2 = run_policy(TrustAndDoubtPolicy(seed=7), req, pages, capacity=3)
    assert [e.hit for e in r1.events] == [e.hit for e in r2.events]


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
